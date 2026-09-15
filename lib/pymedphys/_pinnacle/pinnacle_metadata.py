# Copyright (C) 2019 South Western Sydney Local Health District,
# University of New South Wales
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Helpers for surfacing Pinnacle-specific metadata (lock status,
treatment-trial flag) into DICOM ``*Description`` fields.

Two pieces of Pinnacle data are interesting but have no clean DICOM
equivalent:

* ``plan.PlanInfo.PlanLockStatus`` — a free-text audit string of the form
  ``"The plan was locked by <INITIALS>@with user name <username>@at
  <YYYY-MM-DD HH:MM:SS>."``, or empty/missing when the plan is unlocked.
  Lives on the *plan*, so all trials within a plan share the same value.

* ``plan.Trial.UseTrialForTreatment`` — a boolean (0/1) on each trial.
  Reliable for multi-trial plans (exactly one trial gets 1). Defaults to
  0 in single-trial plans even when that trial *is* the clinical one,
  so it cannot be trusted in isolation.

These helpers parse and combine those two signals to:

1. produce a short, viewer-friendly summary string suitable for stamping
   into ``RTPlanDescription`` / ``StructureSetDescription`` /
   ``SeriesDescription``; and
2. classify each trial as ``"clinical"`` or ``"unknown"`` so callers
   (notably the web app) can filter exports without re-implementing the
   combining logic.
"""

import re
import threading
import time

# Maximum length of the DICOM `ST` (Short Text) VR used by RTPlanDescription
# and StructureSetDescription. SeriesDescription (LO) is shorter (64) but
# we truncate per-field at the call site.
_DICOM_ST_MAX = 1024

# Regex for the standard Pinnacle PlanLockStatus audit string. Tolerant of
# minor whitespace variation around the @ separators but strict about the
# overall shape so we don't false-positive on unrelated text.
_LOCK_STATUS_RE = re.compile(
    r"locked\s+by\s+(?P<initials>\S+?)\s*@\s*"
    r"with\s+user\s+name\s+(?P<username>\S+?)\s*@\s*"
    r"at\s+(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})",
    re.IGNORECASE,
)


def parse_lock_status(lock_str):
    """Parse a Pinnacle ``PlanLockStatus`` string.

    Parameters
    ----------
    lock_str : str or None
        The raw value of ``plan.PlanInfo.PlanLockStatus``.

    Returns
    -------
    dict or None
        ``None`` if the plan is unlocked (empty / missing string).
        Otherwise a dict with keys:

        - ``initials``: locker's initials, or ``""`` if unparseable
        - ``username``: locker's username, or ``""`` if unparseable
        - ``timestamp``: ``YYYY-MM-DD HH:MM:SS`` string, or ``""``
        - ``raw``: the original (stripped) lock string, always present

        The dict is non-None whenever the plan is locked, even if the
        audit fields couldn't be extracted — the caller can fall back
        to ``raw`` in that case.
    """
    if not lock_str or not lock_str.strip():
        return None

    raw = lock_str.strip()
    m = _LOCK_STATUS_RE.search(raw)
    if m:
        return {
            "initials": m.group("initials"),
            "username": m.group("username"),
            "timestamp": m.group("timestamp"),
            "raw": raw,
        }
    # Locked but the audit format didn't match — preserve the raw
    # string so the information isn't lost.
    return {"initials": "", "username": "", "timestamp": "", "raw": raw}


def format_lock_summary(lock_info):
    """Render parsed lock info as a compact human-readable string.

    Parameters
    ----------
    lock_info : dict or None
        The output of :func:`parse_lock_status`.

    Returns
    -------
    str
        ``"unlocked"`` if ``lock_info`` is None;
        ``"locked by <initials>/<username> at <timestamp>"`` if all audit
        fields parsed; or the raw string (truncated) as a fallback.
    """
    if lock_info is None:
        return "unlocked"

    if lock_info["initials"] and lock_info["username"] and lock_info["timestamp"]:
        return (
            f"locked by {lock_info['initials']}/{lock_info['username']} "
            f"at {lock_info['timestamp']}"
        )

    # Couldn't parse the structured fields — surface the raw string so
    # the lock info isn't silently dropped. Cap length defensively.
    raw = lock_info["raw"]
    if len(raw) > 200:
        raw = raw[:197] + "..."
    return f"locked: {raw}"


def resolve_lock_status(plan):
    """Locate ``PlanLockStatus`` for a given :class:`PinnaclePlan`.

    Pinnacle stores the lock/approval audit string in the per-plan
    ``Plan_N/plan.PlanInfo`` file. Some older versions or partial
    archives may have it directly on the ``PlanList`` entry in the
    top-level ``Patient`` file instead. We check the per-plan file
    first because that's the authoritative location, and fall back
    to the ``Patient``-file entry only if it isn't present there.

    Parameters
    ----------
    plan : PinnaclePlan
        The plan to read from.

    Returns
    -------
    str
        The raw lock string, or ``""`` if neither source has it.
    """
    # Per-plan plan.PlanInfo file (authoritative)
    from_file = (plan.plan_info_file.get("PlanLockStatus") or "").strip()
    if from_file:
        return from_file
    # Fall back to the Patient file's PlanList entry
    return (plan.plan_info.get("PlanLockStatus") or "").strip()


def is_clinical_trial(plan_info, trial_info, total_trials):
    """Classify a trial as clinical (treatment-bound) or not.

    Strict interpretation: a trial is clinical only when the plan it
    belongs to is locked AND either there is exactly one trial in the
    plan, or this trial has ``UseTrialForTreatment == 1``.

    Rationale: ``PlanLockStatus`` lives on the plan and applies to every
    trial within it, so "is the plan locked" is a necessary condition.
    But within a locked multi-trial plan, only the trial Pinnacle has
    flagged for treatment is the clinical one — the others are
    reference / QA trials that happen to share the locked plan's audit
    record. For single-trial plans, ``UseTrialForTreatment`` is
    unreliable (Pinnacle defaults it to 0), so we fall back to "the
    plan is locked and this is its only trial" to recognise the
    clinical case.

    .. note::
       This dict-based form is kept for unit testing and direct callers
       that already have the lock string in hand. Production callers
       working with a :class:`PinnaclePlan` instance should prefer
       :func:`is_clinical_trial_for_plan`, which knows where to look
       for ``PlanLockStatus`` across Pinnacle versions.

    Parameters
    ----------
    plan_info : dict
        Must contain ``PlanLockStatus`` (or have it absent for unlocked).
        This may come from either ``PinnaclePlan.plan_info`` or
        ``PinnaclePlan.plan_info_file``; the caller is responsible for
        passing the dict that actually contains the lock field.
    trial_info : dict
        A single trial dict from ``PinnaclePlan.trials``.
    total_trials : int
        Number of trials in the parent plan.

    Returns
    -------
    bool
        True if the trial is the clinical / treatment-bound trial.
    """
    if not isinstance(plan_info, dict) or not isinstance(trial_info, dict):
        return False

    is_locked = bool((plan_info.get("PlanLockStatus") or "").strip())
    if not is_locked:
        return False

    if total_trials == 1:
        return True

    return bool(trial_info.get("UseTrialForTreatment", 0))


def classify_trial(plan_info, trial_info, total_trials):
    """Return ``"clinical"`` or ``"unknown"`` for a trial.

    Convenience wrapper around :func:`is_clinical_trial` returning the
    label used in DICOM description suffixes. ``"unknown"`` is used
    rather than ``"reference"`` to avoid implying we know a trial *is*
    a reference/QA trial when in fact we just can't confirm it's
    clinical.
    """
    return (
        "clinical"
        if is_clinical_trial(plan_info, trial_info, total_trials)
        else "unknown"
    )


def build_pinnacle_metadata_suffix(plan_info, trial_info, total_trials):
    """Build the ``"Pinnacle: ..."`` suffix to append to description fields.

    Format: ``"Pinnacle: <lock-summary>; <classification>"``.

    Examples:

    * ``"Pinnacle: locked by TS/saurat at 2021-08-31 11:32:03; clinical"``
    * ``"Pinnacle: unlocked; unknown"``
    * ``"Pinnacle: locked: <unparseable raw>; clinical"`` (parser fallback)

    Parameters
    ----------
    plan_info : dict
    trial_info : dict
    total_trials : int

    Returns
    -------
    str
        The suffix. Always non-empty so callers can append unconditionally.
    """
    lock_info = parse_lock_status(plan_info.get("PlanLockStatus"))
    lock_summary = format_lock_summary(lock_info)
    classification = classify_trial(plan_info, trial_info, total_trials)
    return f"Pinnacle: {lock_summary}; {classification}"


def append_pinnacle_metadata(
    existing_description, plan_info, trial_info, total_trials, max_length=_DICOM_ST_MAX
):
    """Append the Pinnacle metadata suffix to an existing description.

    Preserves whatever the description already held (typically the patient
    comment) and joins with " | " for readability. Truncates to ``max_length``
    so we don't violate DICOM VR constraints — note that DICOM ``ST`` allows
    1024 chars and ``LO`` allows 64; pass the appropriate limit.

    Parameters
    ----------
    existing_description : str or None
        Whatever the field already contained. ``None`` is treated as empty.
    plan_info : dict
    trial_info : dict
    total_trials : int
    max_length : int
        Hard cap on the returned string length.

    Returns
    -------
    str
    """
    suffix = build_pinnacle_metadata_suffix(plan_info, trial_info, total_trials)
    base = (existing_description or "").strip()
    combined = f"{base} | {suffix}" if base else suffix
    if len(combined) > max_length:
        combined = combined[: max_length - 3].rstrip() + "..."
    return combined


# --- PinnaclePlan-aware convenience functions --------------------------------
#
# The dict-based functions above are kept for testability. Production callers
# usually have a PinnaclePlan instance handy and shouldn't need to know which
# of two files holds the lock status — these wrappers handle that lookup.


def is_clinical_trial_for_plan(plan, trial_info):
    """:func:`is_clinical_trial` with automatic ``PlanLockStatus`` lookup.

    Resolves the lock string from ``plan.plan_info_file`` first, falling
    back to ``plan.plan_info`` (the ``PlanList`` entry from the
    ``Patient`` file) if the per-plan ``plan.PlanInfo`` file is absent
    or doesn't carry the field. Synthesises a minimal ``plan_info`` dict
    with just the resolved lock string and dispatches to the dict-based
    :func:`is_clinical_trial`.
    """
    synthetic_plan_info = {"PlanLockStatus": resolve_lock_status(plan)}
    return is_clinical_trial(synthetic_plan_info, trial_info, len(plan.trials))


def classify_trial_for_plan(plan, trial_info):
    """:func:`classify_trial` with automatic ``PlanLockStatus`` lookup."""
    return "clinical" if is_clinical_trial_for_plan(plan, trial_info) else "unknown"


def build_pinnacle_metadata_suffix_for_plan(plan, trial_info):
    """:func:`build_pinnacle_metadata_suffix` with automatic lookup."""
    synthetic_plan_info = {"PlanLockStatus": resolve_lock_status(plan)}
    return build_pinnacle_metadata_suffix(
        synthetic_plan_info, trial_info, len(plan.trials)
    )


def append_pinnacle_metadata_for_plan(
    existing_description, plan, trial_info, max_length=_DICOM_ST_MAX
):
    """:func:`append_pinnacle_metadata` with automatic lookup."""
    suffix = build_pinnacle_metadata_suffix_for_plan(plan, trial_info)
    base = (existing_description or "").strip()
    combined = f"{base} | {suffix}" if base else suffix
    if len(combined) > max_length:
        combined = combined[: max_length - 3].rstrip() + "..."
    return combined


# --- DICOM General Equipment tag stamping ------------------------------------
#
# These helpers set Manufacturer, ManufacturerModelName, SoftwareVersions
# and InstitutionName on a DICOM dataset, composing existing Pinnacle
# values with site-specific suffixes loaded from config.json.
#
# The separator between "Pinnacle value" and "our suffix" defaults to "-"
# so that downstream readers can trivially split on it.

_EQUIPMENT_SEP = "-"


# --- ApprovalStatus derivation ---------------------------------
#
# Pinnacle's PlanLockStatus is the closest analogue to the DICOM
# ApprovalStatus concept: a locked plan has been signed off inside
# Pinnacle.  These helpers map lock → APPROVED (with the reviewer /
# timestamp audit fields where parseable) and unlocked → UNAPPROVED.


def derive_approval_fields(plan):
    """Derive DICOM approval attributes from the Pinnacle lock status.

    Parameters
    ----------
    plan : PinnaclePlan

    Returns
    -------
    dict
        Keys: ``status`` (``"APPROVED"`` / ``"UNAPPROVED"``),
        ``reviewer_name``, ``review_date`` (DICOM DA or ``""``),
        ``review_time`` (DICOM TM or ``""``).
    """
    lock_info = parse_lock_status(resolve_lock_status(plan))
    if lock_info is None:
        return {
            "status": "UNAPPROVED",
            "reviewer_name": "",
            "review_date": "",
            "review_time": "",
        }

    reviewer = lock_info["username"] or lock_info["initials"] or ""
    review_date = ""
    review_time = ""
    if lock_info["timestamp"]:
        # "YYYY-MM-DD HH:MM:SS" → DA "YYYYMMDD", TM "HHMMSS"
        try:
            date_part, time_part = lock_info["timestamp"].split()
            review_date = date_part.replace("-", "")
            review_time = time_part.replace(":", "")
        except ValueError:
            pass

    return {
        "status": "APPROVED",
        "reviewer_name": reviewer,
        "review_date": review_date,
        "review_time": review_time,
    }


# Recognised values for the ``DICOM_EXPORT.APPROVAL_STATUS`` config key.
#
# "UNAPPROVED" (the default) always exports UNAPPROVED regardless of the
# Pinnacle lock state.  "AUTO" restores the lock-derived behaviour, where a
# locked plan exports as APPROVED.
#
# The default is deliberately the conservative one.  A converted plan that
# arrives at an OIS already marked APPROVED can be scheduled and treated
# without a human ever having reviewed the conversion itself, and this
# exporter's own geometry is not yet validated.  Exporting APPROVED is
# therefore something a site opts into once it trusts the pipeline, not
# something it inherits by accident from a Pinnacle lock that attests to a
# different thing entirely.
_APPROVAL_MODES = ("UNAPPROVED", "AUTO")
_DEFAULT_APPROVAL_MODE = "UNAPPROVED"


def resolve_approval_mode(plan):
    """Return the configured approval mode for *plan*.

    Reads ``DICOM_EXPORT.APPROVAL_STATUS`` via
    ``plan.pinnacle.export_cfg``. Unrecognised or absent values fall back
    to ``"UNAPPROVED"``.
    """
    cfg = getattr(getattr(plan, "pinnacle", None), "export_cfg", None) or {}
    mode = str(cfg.get("APPROVAL_STATUS", "") or "").strip().upper()
    if mode in _APPROVAL_MODES:
        return mode
    if mode:
        plan.logger.warning(
            "Unrecognised DICOM_EXPORT.APPROVAL_STATUS value %r; expected one "
            "of %s. Falling back to %s.",
            mode,
            ", ".join(_APPROVAL_MODES),
            _DEFAULT_APPROVAL_MODE,
        )
    return _DEFAULT_APPROVAL_MODE


def apply_approval_status(ds, plan):
    """Set ApprovalStatus (and audit fields) on *ds*.

    ReviewDate / ReviewTime / ReviewerName are Type 2C — required when
    ApprovalStatus is APPROVED or REJECTED — so they are written (possibly
    empty) whenever the status is not UNAPPROVED.
    """
    mode = resolve_approval_mode(plan)
    fields = derive_approval_fields(plan)

    if mode == "UNAPPROVED":
        ds.ApprovalStatus = "UNAPPROVED"
        if fields["status"] == "APPROVED":
            plan.logger.info(
                "Plan is locked in Pinnacle, but ApprovalStatus is exported "
                "as UNAPPROVED because DICOM_EXPORT.APPROVAL_STATUS is %r. "
                "Set it to 'AUTO' to carry the Pinnacle lock through as "
                "APPROVED.",
                _DEFAULT_APPROVAL_MODE,
            )
        return

    ds.ApprovalStatus = fields["status"]
    if fields["status"] != "UNAPPROVED":
        ds.ReviewDate = fields["review_date"]
        ds.ReviewTime = fields["review_time"]
        ds.ReviewerName = fields["reviewer_name"]
    plan.logger.debug(
        "ApprovalStatus derived from PlanLockStatus: %s", fields["status"]
    )


def _join_with_sep(base, suffix, sep=_EQUIPMENT_SEP):
    """Join *base* and *suffix* with *sep*, handling empty parts cleanly.

    * Both present  → ``"base-suffix"``
    * Only base     → ``"base"``
    * Only suffix   → ``"suffix"``
    * Both empty    → ``""``
    """
    base = (base or "").strip()
    suffix = (suffix or "").strip()
    if base and suffix:
        return f"{base}{sep}{suffix}"
    return base or suffix


def apply_equipment_stamps(
    ds, equipment_cfg, pinnacle_model="", pinnacle_sw="", sep=_EQUIPMENT_SEP
):
    """Set the four DICOM General Equipment tags on *ds*.

    Composes each field from a Pinnacle-originated base value and a
    site-specific suffix drawn from *equipment_cfg* (the
    ``config["DICOM_EQUIPMENT"]`` dict).

    For RT objects the callers pass the Pinnacle plan_info values as
    *pinnacle_model* (``ToolType``) and *pinnacle_sw*
    (``PinnacleVersionDescription``).  For synthesised images where
    those values are unavailable the defaults (empty strings) result in
    only the suffix being written.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The dataset to stamp.
    equipment_cfg : dict
        Should contain any/all of the below:
        ``MANUFACTURER_SUFFIX``, ``MODEL_NAME_SUFFIX``,
        ``SOFTWARE_VERSION_SUFFIX``, ``INSTITUTION_SUFFIX``,
        ``STATION_NAME``.
        Missing keys are silently treated as empty strings.
    pinnacle_model : str
        Pinnacle ``ToolType`` value (e.g. ``"Pinnacle3"``).
    pinnacle_sw : str
        Pinnacle ``PinnacleVersionDescription`` (e.g. ``"16.0"``).
    sep : str
        Separator between base and suffix (default ``"-"``).
    """
    if not equipment_cfg:
        # No config supplied — preserve legacy behaviour: leave
        # whatever the caller already wrote on ds untouched.
        return

    pinnacle_mfr = "Philips"
    mfr_suffix = equipment_cfg.get("MANUFACTURER_SUFFIX", "")
    model_suffix = equipment_cfg.get("MODEL_NAME_SUFFIX", "")
    sw_suffix = equipment_cfg.get("SOFTWARE_VERSION_SUFFIX", "")
    institution = equipment_cfg.get("INSTITUTION_SUFFIX", "")

    mfr_val = _join_with_sep(pinnacle_mfr, mfr_suffix, sep)
    if mfr_val:
        ds.Manufacturer = mfr_val

    model_val = _join_with_sep(pinnacle_model, model_suffix, sep)
    if model_val:
        ds.ManufacturerModelName = model_val

    sw_val = _join_with_sep(pinnacle_sw, sw_suffix, sep)
    if sw_val:
        ds.SoftwareVersions = [sw_val]

    if institution:
        ds.InstitutionName = institution

    # StationName (Type 3): site-configured value, not composed with any
    # Pinnacle base — Pinnacle archives don't carry a meaningful station
    # name for the export host.
    station = (equipment_cfg.get("STATION_NAME") or "").strip()
    if station:
        ds.StationName = station


# --- Pinn2Dicom UID generation -----------------------------------------------
#
# Implements the custom UID algorithm:
#   UID = uid_root.secs.clocks.counts.modality
#
# * secs     — seconds from 01/01/1970 (Unix epoch)
# * clocks   — clock ticks (microseconds since process start) mod 10^5
# * counts   — call counter for the application instance's lifetime mod 1000
# * modality — zero-padded 3-digit index per object type
#
# The generator is a module-level singleton so the counter persists across
# all PinnacleExport / PinnaclePlan instances created during a single run
# of the application (web server or CLI).

# Modality indices — arbitrary but fixed per DICOM object type.
#
# Layout: CT first, then the RT objects in dependency order.  Series UIDs
# mirror their instance index with a leading 1 (instance n -> series 1n),
# so the relationship stays readable at a glance.
UID_MODALITY_INDEX = {
    # SOP Instance UIDs
    "ct": "1",  # CT SOP Instance UID (per slice)
    "struct": "2",  # RTSTRUCT
    "plan": "3",  # RTPLAN
    "dose": "4",  # RTDOSE
    # Study / Frame of Reference (shared across objects)
    "study": "5",  # StudyInstanceUID
    "frame": "6",  # FrameOfReferenceUID
    # Series Instance UIDs (instance index + 10)
    "series_ct": "11",
    "series_struct": "12",
    "series_plan": "13",
    "series_dose": "14",
}


class _UIDGenerator:
    """Thread-safe UID generator following the Pinn2Dicom algorithm.

    A single instance lives at module level (``_uid_generator``).  The
    public entry point is :func:`generate_pinn2dicom_uid`.

    Example output::

        1.2.826.0.1.3680043.10.1361.1766177687.37.32.003
    """

    _MAX_UID_LENGTH = 64  # DICOM PS3.5 §9.1

    def __init__(self):
        self._counter = 0
        self._lock = threading.Lock()
        # Reference point for clock-tick component — set once at import.
        self._start_ns = time.perf_counter_ns()

    def generate(self, uid_root, modality_key):
        """Return a new UID string.

        Parameters
        ----------
        uid_root : str
            Registered DICOM UID root, e.g.
            ``"1.2.826.0.1.3680043.10.1361"``.
        modality_key : str
            One of the keys in :data:`UID_MODALITY_INDEX`.
        """
        modality_index = UID_MODALITY_INDEX[modality_key]

        with self._lock:
            self._counter += 1
            counts = self._counter % 1000

        secs = int(time.time())
        # Microseconds elapsed since process start, capped to 5 digits.
        clocks = ((time.perf_counter_ns() - self._start_ns) // 1000) % 100000

        uid = f"{uid_root}.{secs}.{clocks}.{counts}.{modality_index}"

        if len(uid) > self._MAX_UID_LENGTH:
            raise ValueError(
                f"Generated UID exceeds {self._MAX_UID_LENGTH} chars "
                f"({len(uid)}): {uid}"
            )

        # Guard against non-conformant components (PS3.5 §9.1): digits only,
        # no empty component, and no leading zero on a multi-digit component.
        # This catches a malformed UID_ROOT or a future edit to
        # UID_MODALITY_INDEX before the value ever reaches a DICOM file.
        for component in uid.split("."):
            if not component.isdigit():
                raise ValueError(
                    f"Generated UID has a non-numeric or empty component "
                    f"'{component}': {uid}"
                )
            if len(component) > 1 and component[0] == "0":
                raise ValueError(
                    f"Generated UID component '{component}' has a leading "
                    f"zero, which is invalid in a DICOM UID: {uid}"
                )

        return uid


# Module-level singleton — call counter persists for the lifetime of the
# application process (Flask/waitress server or CLI invocation).
_uid_generator = _UIDGenerator()


def generate_pinn2dicom_uid(uid_root, modality_key):
    """Generate a single DICOM UID using the Pinn2Dicom algorithm.

    Parameters
    ----------
    uid_root : str
        The registered DICOM UID root
        (e.g. ``"1.2.826.0.1.3680043.10.1361"``).
    modality_key : str
        One of ``"struct"``, ``"plan"``, ``"dose"``,
        ``"series_struct"``, ``"series_plan"``, ``"series_dose"``.

    Returns
    -------
    str
        A valid DICOM UID of the form
        ``uid_root.secs.clocks.counts.modality_index``.
    """
    return _uid_generator.generate(uid_root, modality_key)
