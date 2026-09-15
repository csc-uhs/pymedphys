# Copyright (C) 2019 South Western Sydney Local Health District,
# University of New South Wales

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This work is derived from:
# https://github.com/AndrewWAlexander/Pinnacle-tar-DICOM
# which is released under the following license:

# Copyright (c) [2017] [Colleen Henschel, Andrew Alexander]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import re
import time

from pymedphys._imports import pydicom
from pymedphys._pinnacle.pinnacle_exceptions import (
    IsocenterNotFoundError,
    MachineDataNotFoundError,
    MissingCTImageError,
    MissingTrialBeamsError,
)

from .constants import (
    GImplementationClassUID,
    GTransferSyntaxUID,
    RTPLANModality,
    RTPlanSOPClassUID,
    RTStructSOPClassUID,
)
from .pinnacle_metadata import (
    append_pinnacle_metadata_for_plan,
    apply_approval_status,
    apply_equipment_stamps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_dataset():
    """Shorthand for creating a new empty DICOM Dataset."""
    return pydicom.dataset.Dataset()


def _new_sequence():
    """Shorthand for creating a new empty DICOM Sequence."""
    return pydicom.sequence.Sequence()


def _sanitize_for_filename(name):
    """Make a trial name safe for use in a DICOM file name."""
    return re.sub(r"[^\w\-.]", "_", str(name)) if name else "trial"


# DICOM LO (Long String) maximum length — used for SeriesDescription.
_DICOM_LO_MAX = 64

# DICOM SH (Short String) maximum length — used for RTPlanLabel.
_DICOM_SH_MAX = 16


def _truncate_sh(value, logger=None, tag=""):
    """Truncate a value to the DICOM SH (Short String) limit of 16 chars.

    Pinnacle plan names regularly exceed 16 characters (e.g.
    "CopyOf_1_LtBreast" -> RTPlanLabel "CopyOf_1_LtBreast.0" = 19), which
    pydicom warns about and which some PACS reject outright.  The full,
    untruncated name is always still available in the LO-VR fields
    (RTPlanName / StructureSetName / SeriesDescription), so nothing is
    lost — only the short label is clipped.
    """
    text = str(value or "")
    if len(text) <= _DICOM_SH_MAX:
        return text
    truncated = text[:_DICOM_SH_MAX]
    if logger is not None:
        logger.warning(
            "%s value %r is %d chars, exceeding the DICOM SH limit of %d; "
            "truncated to %r (the full name is retained in the "
            "corresponding LO-VR attribute).",
            tag or "SH", text, len(text), _DICOM_SH_MAX, truncated,
        )
    return truncated


def _format_ds(value):
    """Format a float for a DICOM DS (Decimal String, max 16 chars)."""
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if text in ("", "-"):
        text = "0"
    return text[:16]


def _append_trial_series_description(ds, trial_name, prefix):
    """Append 'prefix: trial' to SeriesDescription, respecting the LO VR.

    Exported objects carry their trial name so that a reviewer
    can distinguish trials within a plan at the PACS/TPS end.
    """
    label = f"{prefix}: {trial_name}"
    base = (getattr(ds, "SeriesDescription", "") or "").strip()
    combined = f"{base} - {label}" if base else label
    if len(combined) > _DICOM_LO_MAX:
        combined = combined[: _DICOM_LO_MAX - 3].rstrip() + "..."
    ds.SeriesDescription = combined


def _iter_machine_dicts(node, _depth=0):
    """Yield every dict in *node* that looks like a Pinnacle machine entry.

    ``pinn_to_dict`` nests top-level Pinnacle objects under their type name
    — ``plan.Trial`` parses to ``{"Trial": {...}}``, and
    ``plan.Pinnacle.Machines`` likewise wraps each machine (single machine,
    a list of machines, or a keyed container, depending on the file).  The
    previous implementation only looked at the top level, so
    ``machine.get("Name")`` returned ``None`` and every lookup failed —
    which in turn blocked SAD, DosePerMuAtCalibration and (since the MLC
    fallback was removed) the entire RTPLAN export.

    Rather than hard-coding one nesting shape, this walks the parsed
    structure and yields any dict carrying a "Name" key alongside at least
    one machine-ish attribute.  Depth is bounded so a pathological file
    cannot cause runaway recursion.
    """
    if _depth > 6:
        return

    if isinstance(node, dict):
        keys = set(node)
        if "Name" in keys and keys & {
            "PhotonEnergyList",
            "ElectronEnergyList",
            "MultiLeaf",
            "MultiLeafLayout",
            "VersionTimestamp",
            "SourceToAxisDistance",
            "SourceAxisDistance",
            "SAD",
        }:
            yield node
        for value in node.values():
            yield from _iter_machine_dicts(value, _depth + 1)

    elif isinstance(node, list):
        for item in node:
            yield from _iter_machine_dicts(item, _depth + 1)


def _select_machine(machine_info, machinename, machineversion, logger=None):
    """Return the machine dict matching *machinename*/*machineversion*.

    Searches the parsed ``plan.Pinnacle.Machines`` structure at any nesting
    depth (see :func:`_iter_machine_dicts`).  Returns ``None`` when no match
    is found — callers must fall back to safe behaviour (and, for the MLC
    boundary table, refuse the export rather than assume geometry).
    """
    if machine_info is None:
        return None

    candidates = list(_iter_machine_dicts(machine_info))
    if not candidates:
        if logger is not None:
            logger.warning(
                "No machine entries could be located in the parsed "
                "plan.Pinnacle.Machines structure (top-level keys: %s). The "
                "file may use an unexpected layout.",
                sorted(machine_info)[:10] if isinstance(machine_info, dict)
                else type(machine_info).__name__,
            )
        return None

    # 1. Exact match on both name and version timestamp.
    for machine in candidates:
        if machine.get("Name") == machinename and (
            not machineversion
            or machine.get("VersionTimestamp") == machineversion
        ):
            return machine

    # 2. Name-only match (version timestamps occasionally differ between the
    #    Trial reference and the Machines file).
    for machine in candidates:
        if machine.get("Name") == machinename:
            if logger is not None:
                logger.debug(
                    "Machine '%s' matched by name only (version '%s' not "
                    "matched exactly; file has '%s').",
                    machinename, machineversion,
                    machine.get("VersionTimestamp"),
                )
            return machine

    # No match — report what IS in the file so the mismatch is diagnosable
    # without hand-parsing the archive.
    if logger is not None:
        available = [
            f"{m.get('Name')!r} (version {m.get('VersionTimestamp')!r})"
            for m in candidates[:8]
        ]
        logger.warning(
            "Machine '%s' (version '%s') not found among the %d machine "
            "entry/entries in plan.Pinnacle.Machines. Available: %s",
            machinename, machineversion, len(candidates),
            "; ".join(available) or "(none)",
        )
    return None


def _get_machine_sad_mm(machine, logger, beam_name):
    """Read the source-axis distance (mm) from Pinnacle machine data.

    Pinnacle stores geometry in cm; several key spellings are probed.
    Returns ``None`` when the value is absent or fails a
    sanity check, in which case the caller falls back to 1000 mm with a
    logged assumption.
    """
    if not isinstance(machine, dict):
        return None

    for key in ("SourceToAxisDistance", "SourceAxisDistance", "SAD"):
        raw = machine.get(key)
        if raw in (None, ""):
            continue
        try:
            sad_mm = float(raw) * 10  # Pinnacle cm → DICOM mm
        except (TypeError, ValueError):
            continue
        # Sanity window: clinical linac SADs sit comfortably within
        # 500–3000 mm.  Anything outside suggests a unit/parse problem.
        if 500 <= sad_mm <= 3000:
            return sad_mm
        logger.warning(
            "Beam '%s': machine SAD value %s (key '%s') is outside the "
            "plausible 500–3000 mm range after cm→mm conversion; ignoring "
            "and falling back to 1000 mm.",
            beam_name,
            sad_mm,
            key,
        )
        return None

    return None


def _leaf_boundaries_from_machine(machine, expected_pairs, logger):
    """Build LeafPositionBoundaries (mm strings) from Pinnacle machine data.

    Derives the boundary table from the machine's MultiLeaf layout
    (leaf-pair centre positions and widths, stored in cm) instead
    of assuming the Varian-Millennium pattern.

    Returns a list of ``expected_pairs + 1`` boundary strings, or ``None``
    when the machine data is absent/inconsistent — in which case the
    trial's RTPLAN export fails (MachineDataNotFoundError) rather than
    exporting with an assumed boundary table.
    """
    if not isinstance(machine, dict):
        return None

    multileaf = machine.get("MultiLeaf") or machine.get("MultiLeafLayout")
    if not isinstance(multileaf, dict):
        return None

    pair_container = (
        multileaf.get("LeafPairList")
        or multileaf.get("LeafPairArray")
        or multileaf.get("LeafPairs")
    )
    if pair_container is None:
        return None

    # Normalise the parser output into a flat list of leaf-pair dicts.
    if isinstance(pair_container, dict):
        pairs = [v for v in pair_container.values() if isinstance(v, dict)]
        # Some parses nest each pair under a repeated "LeafPair" key that
        # collapses to a single dict/list — handle a list value too.
        if not pairs:
            inner = pair_container.get("LeafPair")
            if isinstance(inner, list):
                pairs = [p for p in inner if isinstance(p, dict)]
            elif isinstance(inner, dict):
                pairs = [inner]
    elif isinstance(pair_container, list):
        pairs = [p for p in pair_container if isinstance(p, dict)]
    else:
        return None

    geometry = []
    for pair in pairs:
        center = pair.get("YCenterPosition", pair.get("CenterPosition"))
        width = pair.get("Width", pair.get("LeafWidth"))
        if center is None or width is None:
            return None
        try:
            geometry.append((float(center) * 10, float(width) * 10))  # cm → mm
        except (TypeError, ValueError):
            return None

    # Leaf *positions* are read from the control point in LeafPairList's
    # stored order, so that order has to already be ascending in Y for the
    # positions to line up with the boundary table built below.  Sorting
    # the table without checking would quietly pair leaf 1's position with
    # leaf 80's boundary.
    if geometry != sorted(geometry, key=lambda cw: cw[0]):
        logger.warning(
            "Machine MultiLeaf LeafPairList is not stored in ascending "
            "Y order. Leaf positions are read in stored order, so the "
            "boundary table cannot be safely matched to them and this "
            "trial's RTPLAN export will fail.",
        )
        return None

    if len(geometry) != expected_pairs:
        logger.warning(
            "Machine MultiLeaf data describes %d leaf pairs but the plan "
            "data implies %d; the machine boundary table cannot be used and "
            "this trial's RTPLAN export will fail.",
            len(geometry),
            expected_pairs,
        )
        return None

    geometry.sort(key=lambda cw: cw[0])

    boundaries = [geometry[0][0] - geometry[0][1] / 2]
    for center, width in geometry:
        boundaries.append(center + width / 2)

    # The table must be strictly increasing to be a valid boundary set.
    for a, b in zip(boundaries, boundaries[1:]):
        if b <= a:
            logger.warning(
                "Machine MultiLeaf boundaries are not strictly increasing "
                "(%s then %s); the machine boundary table cannot be used and "
                "this trial's RTPLAN export will fail.",
                a,
                b,
            )
            return None

    return [_format_ds(b) for b in boundaries]


def _reject_negated_leaf_coordinates(machine, beam_name):
    """Refuse export when the machine negates its leaf coordinates.

    Raises
    ------
    MachineDataNotFoundError
        When ``MultiLeaf.NegateLeafCoordinates`` is set.
    """
    if not isinstance(machine, dict):
        return
    multileaf = machine.get("MultiLeaf") or machine.get("MultiLeafLayout")
    if not isinstance(multileaf, dict):
        return

    raw = multileaf.get("NegateLeafCoordinates")
    if raw in (None, "", 0, "0", False):
        return

    raise MachineDataNotFoundError(
        f"Beam '{beam_name}': the machine's MultiLeaf data sets "
        f"NegateLeafCoordinates to {raw!r}. The Pinnacle-to-DICOM leaf "
        f"bank mapping used here has only been verified against machines "
        f"that leave this flag clear, and applying it to a machine that "
        f"negates its leaf coordinates would mirror every leaf pair "
        f"without any visible sign of error. This trial's RTPLAN export "
        f"is refused."
    )


def _resolve_beam_isocenter(plan, beam):
    """Resolve the isocenter for a specific beam.

    Priority order:

    1. The beam's ``IsocenterName`` from ``plan.Trial``, looked up in
       ``plan.Points`` (case-insensitive).  A named-but-missing point is
       an error: assuming another point would be dangerous.
    2. The plan-level heuristic (``find_iso_center``: PoiInterpretedType,
       iso-like names, CT centre) for archives whose trial data carries
       no isocenter name.

    Raises
    ------
    IsocenterNotFoundError
        When no isocenter can be resolved.
    """
    iso_name = str(beam.get("IsocenterName", "") or "").strip()
    if iso_name:
        for point in plan.points:
            if str(point.get("Name", "")).strip().lower() == iso_name.lower():
                iso = plan.convert_point(point)
                plan.logger.debug(
                    "Beam '%s': isocenter '%s' resolved from plan.Points: %s",
                    beam.get("Name"),
                    iso_name,
                    iso,
                )
                return iso
        raise IsocenterNotFoundError(
            f"Beam '{beam.get('Name')}' references isocenter point "
            f"'{iso_name}' which was not found in plan.Points."
        )

    # No IsocenterName in this archive — fall back to the plan-level
    # heuristic.  plan.iso_center lazily runs find_iso_center, which no
    # longer defaults to an arbitrary first point (returns [] instead).
    iso = plan.iso_center
    if iso is not None and len(iso) >= 3:
        plan.logger.debug(
            "Beam '%s': no IsocenterName in trial data; using plan-level "
            "isocenter heuristic: %s",
            beam.get("Name"),
            iso,
        )
        return iso

    raise IsocenterNotFoundError(
        f"No isocenter could be determined for beam '{beam.get('Name')}': "
        f"the trial data carries no IsocenterName and no isocenter-like "
        f"point exists in plan.Points."
    )


# Tolerance within which a beam's accumulated Pinnacle control-point weights
# are treated as "should have been exactly 1.0" and renormalised.
_WEIGHT_NORMALISE_TOL = 1e-2


def _normalise_cumulative_weights(weights, beam_name, logger):
    """Rescale cumulative meterset weights so the final value is exactly 1.0.

    Pinnacle writes per-control-point weights to finite precision, so
    accumulating them in float arithmetic lands on e.g. 0.999998 rather
    than 1.  DICOM does not require FinalCumulativeMetersetWeight to be
    1, but Pinnacle's own export writes 1 and downstream systems compare
    against it, so float noise below *_WEIGHT_NORMALISE_TOL* is scaled
    away.  Scaling is dose-neutral: the delivered fraction at each
    control point is CumulativeMetersetWeight / FinalCumulativeMeterset-
    Weight, which the rescaling leaves unchanged.

    A total further from 1.0 than the tolerance is left untouched and
    warned about — that is a data problem rather than accumulated
    rounding, and rescaling would hide it.

    Returns ``(weights, final_weight)``.
    """
    if not weights:
        return weights, 0.0

    final = weights[-1]
    if final <= 0:
        return weights, final

    if abs(final - 1.0) <= _WEIGHT_NORMALISE_TOL:
        if final != 1.0:
            logger.debug(
                "Beam '%s': cumulative meterset weights totalled %r; "
                "rescaling so FinalCumulativeMetersetWeight is exactly 1.",
                beam_name,
                final,
            )
        return [w / final for w in weights], 1.0

    logger.warning(
        "Beam '%s': cumulative meterset weights total %s, more than %s from "
        "1.0. FinalCumulativeMetersetWeight is set to the actual total "
        "(which keeps the plan DICOM-conformant) rather than rescaled, "
        "because a discrepancy this large indicates a data problem rather "
        "than floating-point accumulation. Verify the Pinnacle weights.",
        beam_name,
        final,
        _WEIGHT_NORMALISE_TOL,
    )
    return weights, final


# Pinnacle key spellings that have been seen carrying a per-control-point
# source-to-surface distance (stored in cm).
_CP_SSD_KEYS = ("SSD", "Ssd", "SourceToSkinDistance", "SourceToSurfaceDistance")


def _cp_ssd_mm(control_point):
    """Return a control point's SSD in mm, or ``None`` when absent.

    SourceToSurfaceDistance (300A,0130) is a *control point* attribute:
    on an arc it changes with gantry angle, which is why Pinnacle's own
    export writes a different value on each control point.  Pinnacle only
    stores a per-control-point SSD in some versions, so ``None`` here
    tells the caller to fall back to the beam-level value.
    """
    if not isinstance(control_point, dict):
        return None
    for key in _CP_SSD_KEYS:
        raw = control_point.get(key)
        if raw in (None, ""):
            continue
        try:
            ssd_mm = float(raw) * 10  # Pinnacle cm → DICOM mm
        except (TypeError, ValueError):
            continue
        if 100 <= ssd_mm <= 2000:
            return ssd_mm
    return None


# Candidate Pinnacle machine key spellings for the source-to-device
# distances reported as SourceToBeamLimitingDeviceDistance (300A,00BA).
# Values are stored in cm.
_BLD_DISTANCE_KEYS = {
    "ASYMX": (
        "SourceToXJawDistance",
        "SourceToJawDistanceX",
        "SourceToLeftRightJawDistance",
        "XJawDistance",
    ),
    "ASYMY": (
        "SourceToYJawDistance",
        "SourceToJawDistanceY",
        "SourceToTopBottomJawDistance",
        "YJawDistance",
    ),
    "MLCX": (
        "SourceToMLCDistance",
        "SourceToLeafDistance",
        "SourceToMultiLeafDistance",
        "MLCDistance",
    ),
}


def _bld_distance_mm(machine, device_type):
    """Source-to-beam-limiting-device distance (mm) from machine data.

    SourceToBeamLimitingDeviceDistance (300A,00BA) is Type 3, so
    ``None`` is a valid outcome and the attribute is simply omitted —
    never guessed.  The machine dict and its MultiLeaf sub-dict are both
    searched because Pinnacle stores the MLC distance alongside the leaf
    layout in some versions.
    """
    if not isinstance(machine, dict):
        return None

    search_scopes = [machine]
    multileaf = machine.get("MultiLeaf") or machine.get("MultiLeafLayout")
    if isinstance(multileaf, dict):
        search_scopes.append(multileaf)

    for key in _BLD_DISTANCE_KEYS.get(device_type, ()):
        for scope in search_scopes:
            raw = scope.get(key)
            if raw in (None, ""):
                continue
            try:
                distance_mm = float(raw) * 10  # Pinnacle cm → DICOM mm
            except (TypeError, ValueError):
                continue
            # A beam limiting device sits between the source and the
            # isocentre; anything outside this window is a unit or parse
            # problem rather than real geometry.
            if 100 <= distance_mm <= 1500:
                return distance_mm
    return None


def _fraction_group_for_prescription(ds, groups, prescription, plan, beam_name):
    """Return (creating if needed) the fraction group for *prescription*.

    A Pinnacle trial can mix prescriptions — a 20# phase plus a
    single-fraction boost, say — and Pinnacle's own export emits one
    FractionGroupSequence item per prescription.  Collapsing every beam
    into a single group loses the fractionation of all but one
    prescription and reports a NumberOfBeams that belongs to no single
    group, so each distinct prescription gets its own group here.
    """
    key = str(prescription.get("Name", "")) if prescription else ""
    group = groups.get(key)
    if group is not None:
        return group

    group = _new_dataset()
    group.FractionGroupNumber = len(groups) + 1
    # Type 2: an empty value is valid when the prescription is unknown.
    group.NumberOfFractionsPlanned = (
        prescription.get("NumberOfFractions", "") if prescription else ""
    )
    group.NumberOfBrachyApplicationSetups = "0"
    group.ReferencedBeamSequence = _new_sequence()
    ds.FractionGroupSequence.append(group)
    groups[key] = group

    plan.logger.debug(
        "Fraction group %d created for prescription %r (%s fractions), "
        "first referenced by beam '%s'.",
        group.FractionGroupNumber,
        key,
        group.NumberOfFractionsPlanned,
        beam_name,
    )
    return group


def _gantry_direction_between(prev_angle, next_angle):
    """Return the DICOM rotation direction from *prev_angle* to *next_angle*.

    Angles are in degrees; the shorter arc decides the
    direction, with the delta normalised into (-180, 180].
    """
    try:
        delta = (float(next_angle) - float(prev_angle) + 180.0) % 360.0 - 180.0
    except (TypeError, ValueError):
        return "NONE"
    if delta > 1e-6:
        return "CW"
    if delta < -1e-6:
        return "CC"
    return "NONE"


def _gantry_directions(cp_data_list, total_cps, beam_flag, beam_name, logger):
    """Per-control-point GantryRotationDirection for a whole beam.

    DICOM records the direction that carries the gantry *from* each
    control point to the next, so each entry looks forward and the final
    control point is always "NONE" -- there is no motion after it.  The
    previous implementation looked backwards, which left the first
    control point dependent on the beam-level GantryIsCW / GantryIsCCW
    flags and wrote a direction on the last control point where Pinnacle
    writes "NONE".

    *beam_flag* is used only in the degenerate case where every angle is
    identical yet Pinnacle claims the gantry rotates, which would
    otherwise silently produce a static beam.
    """
    if total_cps <= 0:
        return []

    last_index = len(cp_data_list) - 1
    directions = []
    for j in range(total_cps):
        if j >= total_cps - 1:
            directions.append("NONE")
            continue
        current = cp_data_list[min(j, last_index)]
        following = cp_data_list[min(j + 1, last_index)]
        directions.append(
            _gantry_direction_between(current["gantry"], following["gantry"])
        )

    if beam_flag != "NONE" and all(d == "NONE" for d in directions):
        logger.warning(
            "Beam '%s': Pinnacle reports the gantry rotates (%s) but every "
            "control point carries the same angle, so no direction could be "
            "derived. Using the beam-level flag on the first control point; "
            "verify the control point data.",
            beam_name,
            beam_flag,
        )
        directions[0] = beam_flag

    return directions


def _parse_mlc_leaf_positions(control_point):
    """Parse MLC leaf positions from a Pinnacle control point dict.

    Pinnacle stores the raw points as (left, right) per leaf pair, in cm,
    in the same order as the machine's ``LeafPairList``.  The machine file
    names the banks explicitly -- ``LeftBankName`` is ``x2`` and
    ``RightBankName`` is ``x1`` -- so Pinnacle's left bank is DICOM's +X
    bank and its right bank is DICOM's -X bank.  This is the same
    mirroring the jaws undergo, for the same reason: Pinnacle names its
    collimation in the room frame while DICOM uses IEC beam limiting
    device coordinates, which are the beam's eye view from the source.

    The previous implementation assigned the banks the other way round
    *and* reversed each of them.  Both were wrong: verified against a
    Pinnacle RTPLAN export, X1 is ``-right * 10`` and X2 is
    ``+left * 10``, each in stored order with no reversal, because
    ``LeafPairList`` is itself stored in ascending Y order and therefore
    already lines up with LeafPositionBoundaries.

    Returns (leafpositions, p_count), where leafpositions is the -X bank
    followed by the +X bank and p_count is the number of raw leaf values.
    """
    points_str = control_point["MLCLeafPositions"]["RawData"]["Points[]"]
    raw_points = points_str.split(",")
    p_count = len(raw_points)

    bank_x1 = []  # -X bank: Pinnacle's "right" bank
    bank_x2 = []  # +X bank: Pinnacle's "left" bank
    for i, p in enumerate(raw_points):
        leafpoint = float(p.strip())
        if i % 2 == 0:
            bank_x2.append(leafpoint * 10)
        else:
            bank_x1.append(-leafpoint * 10)

    return bank_x1 + bank_x2, p_count


def _parse_wedge_info(cp_data, plan_logger):
    """Parse wedge information from a Pinnacle control point.

    Returns a dict with wedge details, or None if no wedge is present.
    Keys: type, angle, name, orientation, count.
    """
    wedge_context = cp_data["WedgeContext"]
    wedge_name_raw = wedge_context["WedgeName"]

    if wedge_name_raw in ("No Wedge", ""):
        plan_logger.debug("No wedge present")
        return None

    info = {"count": 1, "angle": wedge_context["Angle"]}
    orientation_raw = wedge_context["Orientation"]

    if "edw" in wedge_name_raw.lower():
        # Enhanced Dynamic Wedge
        plan_logger.debug("EDW wedge present")
        info["type"] = "DYNAMIC"
        if orientation_raw == "WedgeBottomToTop":
            info["name"] = f"{wedge_name_raw.upper()}{info['angle']}IN"
            info["orientation"] = "0"  # TODO confirm orientation mapping
        elif orientation_raw == "WedgeTopToBottom":
            info["name"] = f"{wedge_name_raw.upper()}{info['angle']}OUT"
            info["orientation"] = "180"
        plan_logger.debug("EDW wedge name = %s", info.get("name"))

    elif "UP" in wedge_name_raw:
        # Standard (Universal/Physical) wedge
        plan_logger.debug("Standard wedge present")
        info["type"] = "STANDARD"
        angle_int = int(info["angle"])

        # Map wedge angle to the machine-specific number suffix
        angle_to_suffix = {15: "30", 30: "30", 45: "20", 60: "15"}
        number_suffix = angle_to_suffix.get(angle_int, "")

        orientation_to_label = {
            "WedgeRightToLeft": ("R", "90"),
            "WedgeLeftToRight": ("L", "270"),
            "WedgeTopToBottom": ("OUT", "180"),
            "WedgeBottomToTop": ("IN", "0"),
        }
        label, dicom_orientation = orientation_to_label.get(orientation_raw, ("", "0"))
        info["name"] = f"W{angle_int}{label}{number_suffix}"
        info["orientation"] = dicom_orientation  # TODO: confirm orientation values
        plan_logger.debug("Standard wedge name = %s", info["name"])
    else:
        # Unknown wedge type — treat as no wedge
        plan_logger.warning("Unrecognised wedge name: %s", wedge_name_raw)
        return None

    return info


def _populate_beam_limiting_device_seq(
    beam_ds, p_count, logger=None, machine_boundaries=None, machine=None
):
    """Populate the BeamLimitingDeviceSequence for a beam.

    Creates entries for ASYMX and ASYMY, plus MLCX when the beam has MLC
    leaf data.  The MLCX boundary table MUST come from the Pinnacle
    machine data (*machine_boundaries*): exporting with an
    assumed table (the old hardcoded Varian-Millennium fallback) is
    dangerous — a wrong table silently shifts every leaf pair — so a
    beam that uses an MLC without derivable machine geometry fails the
    trial's RTPLAN export instead.

    Raises
    ------
    MachineDataNotFoundError
        When p_count > 0 but no consistent boundary table was derived
        from the machine data.
    """
    beam_ds.BeamLimitingDeviceSequence = _new_sequence()

    asymx = _new_dataset()
    asymx.RTBeamLimitingDeviceType = "ASYMX"
    asymx.NumberOfLeafJawPairs = "1"
    _set_bld_distance(asymx, machine, "ASYMX", logger)
    beam_ds.BeamLimitingDeviceSequence.append(asymx)

    asymy = _new_dataset()
    asymy.RTBeamLimitingDeviceType = "ASYMY"
    asymy.NumberOfLeafJawPairs = "1"
    _set_bld_distance(asymy, machine, "ASYMY", logger)
    beam_ds.BeamLimitingDeviceSequence.append(asymy)

    # NumberOfLeafJawPairs has VR=IS (integer); use integer division so we
    # never emit a fractional value such as "60.0".
    num_pairs = p_count // 2

    if num_pairs == 0:
        # Jaw-only beam: no MLCX device entry (valid DICOM without one).
        if logger is not None:
            logger.debug(
                "Beam has no MLC leaf data; MLCX omitted from "
                "BeamLimitingDeviceSequence."
            )
        return

    if machine_boundaries is None or len(machine_boundaries) != num_pairs + 1:
        raise MachineDataNotFoundError(
            f"MLC LeafPositionBoundaries for {num_pairs} leaf pairs could "
            f"not be derived from the Pinnacle machine data "
            f"(plan.Pinnacle.Machines). Exporting with an assumed boundary "
            f"table is unsafe, so this trial's RTPLAN export is refused. "
            f"Verify the machine file is present in the archive and its "
            f"MultiLeaf layout matches the plan's leaf count."
        )

    mlcx = _new_dataset()
    mlcx.RTBeamLimitingDeviceType = "MLCX"
    mlcx.NumberOfLeafJawPairs = num_pairs
    # IS-725: boundaries derived from the Pinnacle machine MultiLeaf data.
    mlcx.LeafPositionBoundaries = machine_boundaries
    _set_bld_distance(mlcx, machine, "MLCX", logger)
    beam_ds.BeamLimitingDeviceSequence.append(mlcx)


def _set_bld_distance(device_ds, machine, device_type, logger):
    """Set SourceToBeamLimitingDeviceDistance when machine data supplies it."""
    distance_mm = _bld_distance_mm(machine, device_type)
    if distance_mm is None:
        if logger is not None:
            logger.debug(
                "%s: no source-to-device distance in the machine data; "
                "SourceToBeamLimitingDeviceDistance (Type 3) omitted.",
                device_type,
            )
        return
    device_ds.SourceToBeamLimitingDeviceDistance = _format_ds(distance_mm)


# Pinnacle stores collimation in cm to a machine-defined number of decimal
# places and rounds to it on export; 2 is the value seen on every machine
# examined so far and is used when the machine does not say.
_DEFAULT_COLLIMATION_DECIMALS = 2


def _collimation_decimals(machine):
    """Millimetre decimal places for the X jaws, Y jaws and MLC.

    Pinnacle's own RTPLAN export rounds collimation to the machine's
    configured precision, so a jaw stored as 5.12285 cm is written as
    -51.2 mm.  Exporting the unrounded value asserts a precision the
    planning system never had and makes every jaw position differ from
    Pinnacle's export in the last few digits.  One decimal place in cm is
    one fewer in mm, hence the subtraction.
    """
    multileaf = {}
    if isinstance(machine, dict):
        multileaf = machine.get("MultiLeaf") or machine.get("MultiLeafLayout")
        multileaf = multileaf if isinstance(multileaf, dict) else {}
    else:
        machine = {}

    def decimals(scope, key):
        raw = scope.get(key)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = _DEFAULT_COLLIMATION_DECIMALS
        if value < 0:
            value = _DEFAULT_COLLIMATION_DECIMALS
        return max(value - 1, 0)

    return {
        "x": decimals(machine, "LeftRightDecimalPlaces"),
        "y": decimals(machine, "TopBottomDecimalPlaces"),
        "mlc": decimals(multileaf, "DecimalPlaces"),
    }


def _create_bld_position_entries(x1, x2, y1, y2, leafpositions, decimals=None):
    """Create the BeamLimitingDevicePositionSequence items for a control point.

    Returns a Sequence containing ASYMX and ASYMY entries, plus an MLCX
    entry when the beam has MLC leaf data (jaw-only beams legitimately
    have no MLCX device).

    *decimals* is the mapping returned by :func:`_collimation_decimals`;
    when omitted the values are written unrounded.
    """
    bld_seq = _new_sequence()
    decimals = decimals or {}

    def position(value, device):
        places = decimals.get(device)
        if places is not None:
            value = round(float(value), places)
        # Formatted through _format_ds rather than handed to pydicom as a
        # raw float: the cm -> mm multiplication leaves binary-float noise
        # that pydicom renders in full (e.g. "-12.668299999999" for
        # -12.6683), which wastes DS characters and makes diffs against
        # Pinnacle's own export unreadable.
        return _format_ds(value)

    asymx = _new_dataset()
    asymx.RTBeamLimitingDeviceType = "ASYMX"
    asymx.LeafJawPositions = [position(x1, "x"), position(x2, "x")]
    bld_seq.append(asymx)

    asymy = _new_dataset()
    asymy.RTBeamLimitingDeviceType = "ASYMY"
    asymy.LeafJawPositions = [position(y1, "y"), position(y2, "y")]
    bld_seq.append(asymy)

    if leafpositions:
        mlcx = _new_dataset()
        mlcx.RTBeamLimitingDeviceType = "MLCX"
        mlcx.LeafJawPositions = [position(v, "mlc") for v in leafpositions]
        bld_seq.append(mlcx)

    return bld_seq


def _collimation_changed(previous, current):
    """True when any jaw or leaf position differs between control points."""
    for key in ("x1", "x2", "y1", "y2"):
        if previous.get(key) != current.get(key):
            return True
    return previous.get("leafpositions") != current.get("leafpositions")


def _set_cp_ssd(cp, cp_entry):
    """Write SourceToSurfaceDistance on a non-first control point.

    Only written when Pinnacle supplies a per-control-point value: on a
    subsequent control point, repeating the beam-level SSD would assert
    that the distance is unchanged, which is not true on an arc.
    """
    ssd_mm = cp_entry.get("ssd")
    if ssd_mm is not None:
        cp.SourceToSurfaceDistance = _format_ds(ssd_mm)


def _create_wedge_position_seq():
    """Create a WedgePositionSequence for a control point with wedge IN."""
    seq = _new_sequence()
    wp = _new_dataset()
    wp.WedgePosition = "IN"
    wp.ReferencedWedgeNumber = "1"
    seq.append(wp)
    return seq


def _create_wedge_sequence(wedge_info):
    """Create the beam-level WedgeSequence from parsed wedge info."""
    seq = _new_sequence()
    wedge = _new_dataset()
    wedge.WedgeNumber = 1
    wedge.WedgeType = wedge_info["type"]
    wedge.WedgeAngle = wedge_info["angle"]
    wedge.WedgeID = wedge_info["name"]
    wedge.WedgeOrientation = wedge_info["orientation"]
    # WedgeFactor (Type 3, VR=DS) is omitted rather than written as "": an
    # empty string is not a valid Decimal String. Populate with the real
    # factor from Pinnacle data when it becomes available.
    seq.append(wedge)
    return seq


def _populate_first_control_point(
    cp,
    beam_ds,
    beam,
    plan,
    beam_energy,
    doserate,
    gantryrotdir,
    numwedges,
    cp_entry,
    iso_center,
    decimals=None,
):
    """Populate all attributes required by DICOM for the first control point.

    Per DICOM C.8.8.14.5: at the first control point, ALL applicable
    attributes must be present. This includes 1C and 2C attributes.

    All geometric values now come from *cp_entry* — the parsed data of
    the beam's first Pinnacle control point — rather than values
    frozen from whichever control point was parsed last.
    """
    # --- Required energy and dose rate (Type 3 but universally expected) ---
    cp.NominalBeamEnergy = beam_energy
    cp.DoseRateSet = doserate

    # --- Gantry (1C — required at first CP) ---
    cp.GantryAngle = cp_entry["gantry"]
    cp.GantryRotationDirection = gantryrotdir

    # --- Collimator (1C — required at first CP) ---
    cp.BeamLimitingDeviceAngle = cp_entry["collimator"]
    cp.BeamLimitingDeviceRotationDirection = "NONE"

    # --- Patient Support / Couch (1C — required at first CP) ---
    cp.PatientSupportAngle = cp_entry["couch"]
    cp.PatientSupportRotationDirection = "NONE"

    # --- Table Top Eccentric (1C — required at first CP) ---
    cp.TableTopEccentricAngle = "0"
    cp.TableTopEccentricRotationDirection = "NONE"

    # --- Table Top Position (2C — required at first CP, may be empty) ---
    cp.TableTopVerticalPosition = ""
    cp.TableTopLongitudinalPosition = ""
    cp.TableTopLateralPosition = ""

    # --- Isocenter (2C — required at first CP when isocentric) ---
    # Resolved per beam from the trial's IsocenterName.
    cp.IsocenterPosition = iso_center

    # --- Source to Surface Distance (Type 3) ---
    # This is a control point attribute, not a beam attribute: on an arc it
    # tracks the gantry angle.  Use the control point's own value when
    # Pinnacle stores one, otherwise the beam-level SSD.
    ssd_mm = cp_entry.get("ssd")
    if ssd_mm is None:
        try:
            ssd_mm = float(beam["SSD"]) * 10
        except (KeyError, TypeError, ValueError):
            ssd_mm = None
    if ssd_mm is not None:
        cp.SourceToSurfaceDistance = _format_ds(ssd_mm)

    # --- Wedge position (1C — required when wedges present) ---
    if numwedges > 0:
        cp.WedgePositionSequence = _create_wedge_position_seq()

    # --- Beam Limiting Device positions (1C) ---
    cp.BeamLimitingDevicePositionSequence = _create_bld_position_entries(
        cp_entry["x1"],
        cp_entry["x2"],
        cp_entry["y1"],
        cp_entry["y2"],
        cp_entry["leafpositions"],
        decimals,
    )

    # --- Beam-level counts (Type 1 — placed here for locality but belong to beam) ---
    beam_ds.NumberOfWedges = numwedges
    beam_ds.NumberOfCompensators = "0"
    beam_ds.NumberOfBoli = "0"
    beam_ds.NumberOfBlocks = "0"


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def convert_plan(plan, export_path):
    """Export RTPLAN files for every trial in the plan.

    For each trial: switch the plan's active trial, generate fresh per-trial
    UIDs, and call ``convert_plan_for_trial`` to write a single RTPLAN file.
    The matching RTSTRUCT UID for that trial is also generated here so the
    written RTPLAN's ``ReferencedStructureSetSequence`` lines up with the
    RTSTRUCT that ``convert_struct`` will emit for the same trial.
    """
    if not plan.primary_image:
        plan.logger.error("No primary image found for plan. Unable to generate RTPLAN.")
        raise MissingCTImageError("Plan has no primary image associated with it.")

    # TODO Test the RTPLAN export functionality and remove this warning
    plan.logger.warning(
        "RTPLAN export functionality is currently not validated and not stable. "
        "Use with caution."
    )

    for trial_info in plan.trials:
        plan.active_trial = trial_info["Name"]
        plan.logger.info("Exporting RTPLAN for trial: %s", trial_info["Name"])

        uids = plan.generate_uids_for_trial(trial_info)
        try:
            convert_plan_for_trial(
                plan,
                trial_info,
                plan_instance_uid=uids["plan"],
                struct_instance_uid=uids["struct"],
                series_instance_uid=uids["series_plan"],
                export_path=export_path,
            )
        except MissingTrialBeamsError as exc:
            plan.logger.warning(
                "Skipping RTPLAN for trial '%s': %s", trial_info["Name"], exc
            )
            continue
        except (IsocenterNotFoundError, MachineDataNotFoundError) as exc:
            # Refusing to guess geometry; surface
            # loudly so the operator knows this trial's RTPLAN was refused.
            plan.logger.error(
                "RTPLAN export failed for trial '%s': %s", trial_info["Name"], exc
            )
            continue


# ---------------------------------------------------------------------------
# Per-trial RTPLAN generation
# ---------------------------------------------------------------------------


def convert_plan_for_trial(
    plan,
    trial_info,
    plan_instance_uid,
    struct_instance_uid,
    series_instance_uid,
    export_path,
):
    """Write a single RTPLAN DICOM file for one specific trial."""

    patient_info = plan.pinnacle.patient_info
    plan_info = plan.plan_info
    image_info = plan.primary_image.image_info[0]
    machine_info = plan.machine_info
    patient_position = plan.patient_position

    # --- File meta ---
    file_meta = _new_dataset()
    file_meta.MediaStorageSOPClassUID = RTPlanSOPClassUID
    file_meta.TransferSyntaxUID = GTransferSyntaxUID
    file_meta.MediaStorageSOPInstanceUID = plan_instance_uid
    file_meta.ImplementationClassUID = GImplementationClassUID

    safe_trial = _sanitize_for_filename(trial_info.get("Name"))
    rp_filename = f"RP.{safe_trial}.{plan_instance_uid}.dcm"
    ds = pydicom.dataset.FileDataset(
        rp_filename, {}, file_meta=file_meta, preamble=b"\x00" * 128
    )

    # --- Study / Patient level ---
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.InstanceCreationDate = time.strftime("%Y%m%d")
    ds.InstanceCreationTime = time.strftime("%H%M%S")
    ds.SOPClassUID = RTPlanSOPClassUID
    ds.SOPInstanceUID = plan_instance_uid

    datetimesplit = plan_info["ObjectVersion"]["WriteTimeStamp"].split()
    if trial_info:
        datetimesplit = trial_info["ObjectVersion"]["WriteTimeStamp"].split()

    ds.StudyDate = datetimesplit[0].replace("-", "")
    ds.StudyTime = datetimesplit[1].replace(":", "")
    ds.AccessionNumber = ""
    ds.Modality = RTPLANModality
    ds.Manufacturer = ""  # Type 2; overwritten by apply_equipment_stamps
    ds.OperatorsName = ""
    ds.ManufacturerModelName = plan_info.get("ToolType", "")
    ds.SoftwareVersions = [plan_info["PinnacleVersionDescription"]]

    # Apply site-specific equipment identification stamps from config
    apply_equipment_stamps(
        ds,
        plan.pinnacle.equipment_cfg,
        pinnacle_model=plan_info.get("ToolType", ""),
        pinnacle_sw=plan_info.get("PinnacleVersionDescription", ""),
    )
    ds.PhysiciansOfRecord = patient_info["RadiationOncologist"]
    ds.PatientName = patient_info["FullName"]
    ds.PatientBirthDate = patient_info["DOB"]
    ds.PatientID = patient_info["MedicalRecordNumber"]
    ds.PatientSex = patient_info.get("Gender", "")[:1]
    ds.StudyInstanceUID = image_info["StudyInstanceUID"]
    ds.SeriesInstanceUID = series_instance_uid
    ds.StudyID = plan.primary_image.image["StudyID"]
    ds.FrameOfReferenceUID = image_info["FrameUID"]
    ds.PositionReferenceIndicator = ""

    # Carry the trial name so reviewers can distinguish
    # trials within a plan at the PACS/TPS end.
    _append_trial_series_description(ds, trial_info.get("Name", ""), "Plan")

    # --- Plan identification ---
    # RTPlanLabel is VR SH (max 16 chars); RTPlanName is LO (max 64) and
    # keeps the full untruncated Pinnacle plan name.
    ds.RTPlanLabel = _truncate_sh(
        f"{plan_info['PlanName']}.0", plan.logger, "RTPlanLabel")
    ds.RTPlanName = plan_info["PlanName"]
    ds.RTPlanDescription = append_pinnacle_metadata_for_plan(
        None,
        plan,
        trial_info,
        max_length=1024,
    )
    ds.RTPlanDate = ds.StudyDate
    ds.RTPlanTime = ds.StudyTime
    ds.PlanIntent = ""  # Type 3 — no curative/palliative source in Pinnacle data
    # PATIENT is correct because this RTPLAN always carries a
    # ReferencedStructureSetSequence pointing at an image-based RTSTRUCT
    # (the exporter refuses to run without a primary CT image).
    ds.RTPlanGeometry = "PATIENT"

    # --- Referenced Structure Set ---
    ds.ReferencedStructureSetSequence = _new_sequence()
    ref_struct = _new_dataset()
    ref_struct.ReferencedSOPClassUID = RTStructSOPClassUID
    ref_struct.ReferencedSOPInstanceUID = struct_instance_uid
    ds.ReferencedStructureSetSequence.append(ref_struct)

    # Derived from Pinnacle PlanLockStatus (locked → APPROVED
    # with reviewer/timestamp audit fields, unlocked → UNAPPROVED).
    apply_approval_status(ds, plan)

    # --- Fraction Groups ---
    # One FractionGroupSequence item per distinct prescription referenced by
    # the trial's beams, matching Pinnacle's own export.  Groups are created
    # lazily as beams resolve their prescription.
    ds.FractionGroupSequence = _new_sequence()
    fraction_groups = {}

    # --- Sequences that are populated per-beam ---
    ds.BeamSequence = _new_sequence()
    ds.PatientSetupSequence = _new_sequence()
    patient_setups = {}  # PatientPosition -> PatientSetupNumber

    beam_count = 0

    beam_list = trial_info["BeamList"] if trial_info["BeamList"] else []
    if len(beam_list) == 0:
        plan.logger.warning("No Beams found in Trial. Unable to generate RTPLAN.")
        raise MissingTrialBeamsError("No Beams found in Trial.")

    # =======================================================================
    # BEAM LOOP
    # =======================================================================
    for beam in beam_list:
        beam_count += 1
        plan.logger.info("Exporting Plan for beam: %s", beam["Name"])

        # Meterset weights are per-beam: reset here so beam N never inherits
        # cumulative weights parsed from an earlier beam.
        metersetweight = ["0"]

        # --- Patient Setup (shared by beams in the same position) ---
        # Pinnacle's own export writes one setup item and points every beam
        # at it.  Emitting a duplicate per beam is legal but implies the
        # patient is repositioned between beams that in fact share a setup.
        setup_number = patient_setups.get(patient_position)
        if setup_number is None:
            setup_number = len(patient_setups) + 1
            patient_setups[patient_position] = setup_number
            patient_setup = _new_dataset()
            patient_setup.PatientPosition = patient_position
            patient_setup.PatientSetupNumber = setup_number
            ds.PatientSetupSequence.append(patient_setup)

        # --- Referenced Beam ---
        # Appended to its prescription's fraction group further down, once
        # the beam's prescription has been resolved.
        ref_beam = _new_dataset()
        ref_beam.ReferencedBeamNumber = beam_count

        # --- Beam dataset ---
        beam_ds = _new_dataset()
        ds.BeamSequence.append(beam_ds)

        beam_ds.Manufacturer = ds.Manufacturer  # Consistent with plan-level stamp
        beam_ds.BeamNumber = beam_count
        beam_ds.TreatmentDeliveryType = "TREATMENT"
        beam_ds.ReferencedPatientSetupNumber = setup_number
        # SourceAxisDistance is overridden below from the
        # Pinnacle machine data once the beam's machine has been resolved;
        # 1000 mm remains only as a logged fallback.
        beam_ds.SourceAxisDistance = "1000"
        beam_ds.PrimaryDosimeterUnit = "MU"

        # Primary Fluence Mode
        beam_ds.PrimaryFluenceModeSequence = _new_sequence()
        fluence = _new_dataset()
        fluence.FluenceMode = "STANDARD"
        beam_ds.PrimaryFluenceModeSequence.append(fluence)

        beam_ds.BeamName = beam["FieldID"]
        beam_ds.BeamDescription = beam["Name"]

        # Radiation type
        modality = beam["Modality"]
        if "Photons" in modality:
            beam_ds.RadiationType = "PHOTON"
        elif "Electrons" in modality:
            beam_ds.RadiationType = "ELECTRON"
        else:
            plan.logger.warning(
                "Beam '%s': unrecognised modality '%s'; RadiationType left "
                "empty (proton/ion plans are not yet supported).",
                beam["Name"],
                modality,
            )
            beam_ds.RadiationType = ""

        # Beam type
        if "STATIC" in beam["SetBeamType"].upper():
            beam_ds.BeamType = beam["SetBeamType"].upper()
        else:
            beam_ds.BeamType = "DYNAMIC"

        beam_ds.TreatmentMachineName = beam["MachineNameAndVersion"].partition(":")[0]

        # --- Isocenter (per beam) ---
        # Resolved from the trial's IsocenterName where available; raises
        # IsocenterNotFoundError (failing this trial's RTPLAN) rather than
        # assuming an arbitrary point.
        beam_iso_center = _resolve_beam_isocenter(plan, beam)

        # --- Dose Reference Point ---
        doserefpt = None
        for point in plan.points:
            if point["Name"] == beam["PrescriptionPointName"]:
                doserefpt = plan.convert_point(point)
                plan.logger.debug("Dose reference point found: %s", point["Name"])

        if not doserefpt:
            plan.logger.debug("No dose reference point, setting to isocenter")
            doserefpt = beam_iso_center

        plan.logger.debug("Dose reference point: %s", doserefpt)
        ref_beam.BeamDoseSpecificationPoint = doserefpt

        # --- Control Point Manager ---
        beam_ds.ControlPointSequence = _new_sequence()

        cp_manager = beam["CPManager"]
        if "CPManagerObject" in cp_manager:
            cp_manager = cp_manager["CPManagerObject"]

        numctrlpts = cp_manager["NumberOfControlPoints"]
        plan.logger.debug("Number of control points: %s", numctrlpts)

        # --- Parse control point data from Pinnacle ---
        # Every Pinnacle control point is parsed into its own entry so the
        # builders can give each DICOM control point its own jaw and MLC
        # positions and mechanical angles — previously only the last CP's
        # values survived the loop and were reused for every DICOM CP.
        cp_data_list = []
        for cp_data in cp_manager["ControlPointList"]:
            metersetweight.append(cp_data["Weight"])

            leafpositions, p_count = _parse_mlc_leaf_positions(cp_data)

            # Pinnacle names its jaws in the room/patient frame; DICOM
            # BeamLimitingDevicePositionSequence uses IEC beam limiting
            # device coordinates, which are the beam's eye view *from the
            # source* and therefore mirrored relative to Pinnacle on both
            # axes.  X1 is the -X jaw and X2 the +X jaw, so X1 comes from
            # RightJawPosition and X2 from LeftJawPosition; likewise Y1
            # from TopJawPosition and Y2 from BottomJawPosition.  The
            # previous mapping produced jaw pairs whose magnitudes were
            # swapped relative to Pinnacle's own RTPLAN export (e.g.
            # -45\55 where Pinnacle wrote -55\45), which silently
            # mirrors every asymmetric field.
            cp_data_list.append(
                {
                    "x1": -cp_data["RightJawPosition"] * 10,
                    "x2": cp_data["LeftJawPosition"] * 10,
                    "y1": -cp_data["TopJawPosition"] * 10,
                    "y2": cp_data["BottomJawPosition"] * 10,
                    "leafpositions": leafpositions,
                    "p_count": p_count,
                    "gantry": cp_data["Gantry"],
                    "collimator": cp_data["Collimator"],
                    "couch": cp_data["Couch"],
                    "ssd": _cp_ssd_mm(cp_data),
                }
            )

        if not cp_data_list:
            plan.logger.warning(
                "Beam '%s' has no control points; skipping beam.", beam["Name"]
            )
            raise MissingTrialBeamsError(
                f"Beam '{beam['Name']}' has no control points."
            )

        # Wedge context is constant across a beam's control points in
        # Pinnacle; read it from the first CP.
        wedge_info = _parse_wedge_info(
            cp_manager["ControlPointList"][0], plan.logger
        )
        p_count = cp_data_list[0]["p_count"]

        numwedges = wedge_info["count"] if wedge_info else 0

        # --- Prescription and energy ---
        # A missing prescription used to raise IndexError and abort the whole
        # export; the beam is still exported, in its own fraction group with
        # an empty (Type 2) NumberOfFractionsPlanned.
        matching = [
            p
            for p in (trial_info.get("PrescriptionList") or [])
            if p.get("Name") == beam.get("PrescriptionName")
        ]
        prescription = matching[0] if matching else None
        if prescription is None:
            plan.logger.warning(
                "Beam '%s': prescription %r not found in the trial's "
                "PrescriptionList (available: %s); NumberOfFractionsPlanned "
                "will be left empty for its fraction group.",
                beam.get("Name"),
                beam.get("PrescriptionName"),
                [p.get("Name") for p in (trial_info.get("PrescriptionList") or [])],
            )

        fraction_group = _fraction_group_for_prescription(
            ds, fraction_groups, prescription, plan, beam.get("Name")
        )
        fraction_group.ReferencedBeamSequence.append(ref_beam)

        mnv = beam["MachineNameAndVersion"]
        if ": " in mnv:
            machinename, machineversion = mnv.split(": ", 1)
        else:
            plan.logger.warning(
                "Beam '%s': MachineNameAndVersion '%s' is not in the expected "
                "'name: version' form; version treated as empty.",
                beam["Name"],
                mnv,
            )
            machinename, machineversion = mnv, ""
        machineenergyname = beam["MachineEnergyName"]

        # --- Machine data ---
        # plan.machine_info may be None when plan.Pinnacle.Machines is
        # missing or unparseable; _select_machine handles that (previously
        # machine_info["Name"] would raise TypeError here).
        machine = _select_machine(
            machine_info, machinename, machineversion, plan.logger
        )
        if machine is None:
            plan.logger.warning(
                "Beam '%s': no machine entry matching '%s' (version '%s'); "
                "SAD falls back to 1000 mm and, if this beam uses an MLC, "
                "the trial's RTPLAN export will fail (no assumed "
                "leaf-boundary table is used). See the preceding message "
                "for the machine names present in the file.",
                beam["Name"],
                machinename,
                machineversion,
            )

        # SourceAxisDistance from the machine geometry (fallback 1000 mm).
        sad_mm = _get_machine_sad_mm(machine, plan.logger, beam["Name"])
        if sad_mm is not None:
            beam_ds.SourceAxisDistance = _format_ds(sad_mm)
            plan.logger.debug(
                "Beam '%s': SourceAxisDistance %s mm read from machine data.",
                beam["Name"],
                beam_ds.SourceAxisDistance,
            )
        else:
            plan.logger.warning(
                "Beam '%s': SourceAxisDistance not found in machine data; "
                "assuming standard 1000 mm.",
                beam["Name"],
            )

        # Pinnacle can store leaf coordinates with the sign convention
        # inverted.  The bank mapping below was verified only against
        # machines with this flag clear, and honouring it incorrectly would
        # mirror every leaf pair, so an export that would depend on it is
        # refused rather than guessed -- the same stance taken for the
        # leaf boundary table.
        _reject_negated_leaf_coordinates(machine, beam["Name"])

        # MLC LeafPositionBoundaries from the machine MultiLeaf layout.
        machine_boundaries = _leaf_boundaries_from_machine(
            machine, p_count // 2, plan.logger
        )
        if machine_boundaries is not None:
            plan.logger.debug(
                "Beam '%s': LeafPositionBoundaries derived from machine data "
                "(%d boundaries).",
                beam["Name"],
                len(machine_boundaries),
            )

        energy_matches = re.findall(r"[-+]?\d*\.\d+|\d+", machineenergyname)
        if energy_matches:
            beam_energy = energy_matches[0]
        else:
            plan.logger.warning(
                "Beam '%s': could not parse a numeric energy from '%s'; "
                "defaulting NominalBeamEnergy to 0.",
                beam["Name"],
                machineenergyname,
            )
            beam_energy = "0"

        # Find DosePerMuAtCalibration from machine data
        dose_per_mu_at_cal = -1
        if machine is not None:
            for energy in machine.get("PhotonEnergyList", []) or []:
                if energy.get("Name") == machineenergyname:
                    dose_per_mu_at_cal = energy["PhysicsData"]["OutputFactor"][
                        "DosePerMuAtCalibration"
                    ]
                    plan.logger.debug(
                        "Using DosePerMuAtCalibration of: %s", dose_per_mu_at_cal
                    )

        prescripdose = beam["MonitorUnitInfo"]["PrescriptionDose"]
        normdose = beam["MonitorUnitInfo"]["NormalizedDose"]

        if normdose == 0:
            # A zero-dose beam (a setup/verification field) still carries an
            # explicit zero in Pinnacle's own export; BeamDose was previously
            # left unset here while BeamMeterset was written.
            ref_beam.BeamDose = _format_ds(0)
            ref_beam.BeamMeterset = _format_ds(0)
        elif dose_per_mu_at_cal <= 0:
            # No valid calibration was located (machine/energy mismatch, or a
            # non-positive value). Computing prescripdose / (normdose *
            # dose_per_mu_at_cal) here would produce a negative meterset or a
            # division by zero, so leave the Type-3 BeamMeterset/BeamDose unset
            # and warn instead of emitting an invalid value.
            plan.logger.warning(
                "Beam '%s': no valid DosePerMuAtCalibration found (machine "
                "'%s', version '%s', energy '%s'); BeamMeterset and BeamDose "
                "left unset to avoid an invalid value.",
                beam["Name"],
                machinename,
                machineversion,
                machineenergyname,
            )
        else:
            # Formatted rather than assigned as raw floats: the division
            # leaves noise that pydicom renders in full, which can run a
            # DS value up against its 16-character limit.
            ref_beam.BeamDose = _format_ds(prescripdose / 100)
            ref_beam.BeamMeterset = _format_ds(
                prescripdose / (normdose * dose_per_mu_at_cal)
            )

        # Gantry rotation direction
        is_ccw = cp_manager.get("GantryIsCCW") == 1
        is_cw = cp_manager.get("GantryIsCW") == 1
        if is_ccw and is_cw:
            plan.logger.warning(
                "Beam '%s': both GantryIsCCW and GantryIsCW are set; "
                "defaulting GantryRotationDirection to CW.",
                beam["Name"],
            )
        gantryrotdir = "NONE"
        if is_ccw:
            gantryrotdir = "CC"
        if is_cw:
            gantryrotdir = "CW"

        plan.logger.debug("Beam MU: %s", getattr(ref_beam, "BeamMeterset", None))

        doserate = beam.get("DoseRate", 0)

        # ===================================================================
        # Branch: Step & Shoot vs. Non-Step-and-Shoot
        # ===================================================================
        is_step_and_shoot = (
            "STEP" in beam["SetBeamType"].upper()
            and "SHOOT" in beam["SetBeamType"].upper()
        )

        if is_step_and_shoot:
            _build_step_and_shoot_control_points(
                beam_ds,
                beam,
                plan,
                numctrlpts,
                metersetweight,
                beam_energy,
                doserate,
                gantryrotdir,
                numwedges,
                wedge_info,
                cp_data_list,
                p_count,
                machine_boundaries,
                beam_iso_center,
                machine,
            )
        else:
            _build_non_ss_control_points(
                beam_ds,
                beam,
                plan,
                numctrlpts,
                metersetweight,
                beam_energy,
                doserate,
                gantryrotdir,
                numwedges,
                wedge_info,
                cp_data_list,
                p_count,
                machine_boundaries,
                beam_iso_center,
                machine,
            )

        numwedges = 0  # Reset for next beam

    # --- Fraction group summaries ---
    # NumberOfBeams is per group, not the trial-wide beam count.
    for group in ds.FractionGroupSequence:
        group.NumberOfBeams = len(group.ReferencedBeamSequence)
    plan.logger.debug(
        "Trial '%s': %d beam(s) across %d fraction group(s).",
        trial_info.get("Name"),
        beam_count,
        len(ds.FractionGroupSequence),
    )

    # --- Save ---
    output_file = os.path.join(export_path, rp_filename)
    plan.logger.info("Creating Plan file: %s", output_file)
    ds.save_as(output_file, enforce_file_format=True)


# ---------------------------------------------------------------------------
# Step & Shoot control point builder
# ---------------------------------------------------------------------------


def _build_step_and_shoot_control_points(
    beam_ds,
    beam,
    plan,
    numctrlpts,
    metersetweight,
    beam_energy,
    doserate,
    gantryrotdir,
    numwedges,
    wedge_info,
    cp_data_list,
    p_count,
    machine_boundaries,
    iso_center,
    machine=None,
):
    """Build control points for a Step & Shoot beam.

    Pinnacle control point *i* maps to DICOM control points
    ``2i`` and ``2i+1`` (the segment's dose is delivered between the
    pair), and each pair carries that segment's own jaw and MLC
    positions — previously every control point reused the last
    segment's aperture.

    FinalCumulativeMetersetWeight is set to the last control
    point's cumulative weight instead of a hardcoded "1".
    """
    plan.logger.debug("Using Step & Shoot")

    decimals = _collimation_decimals(machine)

    total_cps = numctrlpts * 2
    beam_ds.NumberOfControlPoints = total_cps
    # SourceToSurfaceDistance (300A,0130) is defined on the Control Point
    # Sequence, not on the beam, so it is written per control point below.

    if numwedges > 0:
        beam_ds.WedgeSequence = _create_wedge_sequence(wedge_info)

    # --- Pass 1: cumulative meterset weights per DICOM CP ---
    # Odd control points carry the segment's meterset weight increment.
    cumulative_weights = []
    currentmeterset = 0.0
    metercount = 1
    for j in range(total_cps):
        if j % 2 == 1:
            increment = float(metersetweight[metercount])
            if increment < 0:
                plan.logger.warning(
                    "Beam '%s': negative meterset weight (%s) at Pinnacle "
                    "control point %d; cumulative weights will not be "
                    "monotonic. Verify the plan data.",
                    beam["Name"],
                    increment,
                    metercount,
                )
            currentmeterset += increment
            metercount += 1
        cumulative_weights.append(currentmeterset)

    cumulative_weights, final_weight = _normalise_cumulative_weights(
        cumulative_weights, beam["Name"], plan.logger
    )
    beam_ds.FinalCumulativeMetersetWeight = _format_ds(final_weight)

    # --- Pass 2: build the control points ---
    for j in range(total_cps):
        cp = _new_dataset()
        beam_ds.ControlPointSequence.append(cp)

        cp.ControlPointIndex = j
        cp.CumulativeMetersetWeight = _format_ds(cumulative_weights[j])

        # DICOM CP j belongs to Pinnacle segment j // 2.
        cp_entry = cp_data_list[min(j // 2, len(cp_data_list) - 1)]

        if j == 0:
            # First control point: all attributes must be present (DICOM C.8.8.14.5)
            _populate_first_control_point(
                cp,
                beam_ds,
                beam,
                plan,
                beam_energy,
                doserate,
                gantryrotdir,
                numwedges,
                cp_entry,
                iso_center,
                decimals,
            )
        else:
            # Subsequent control points: this segment's own jaw and MLC
            # positions (jaws can change between segments too).
            cp.BeamLimitingDevicePositionSequence = _create_bld_position_entries(
                cp_entry["x1"],
                cp_entry["x2"],
                cp_entry["y1"],
                cp_entry["y2"],
                cp_entry["leafpositions"],
                decimals,
            )
            _set_cp_ssd(cp, cp_entry)

    # Beam Limiting Device Sequence (beam level)
    _populate_beam_limiting_device_seq(
        beam_ds, p_count, plan.logger, machine_boundaries, machine
    )


# ---------------------------------------------------------------------------
# Non-Step-and-Shoot (conformal arc / dynamic) control point builder
# ---------------------------------------------------------------------------


def _build_non_ss_control_points(
    beam_ds,
    beam,
    plan,
    numctrlpts,
    metersetweight,
    beam_energy,
    doserate,
    gantryrotdir,
    numwedges,
    wedge_info,
    cp_data_list,
    p_count,
    machine_boundaries,
    iso_center,
    machine=None,
):
    """Build control points for a non-Step-and-Shoot beam (e.g. conformal arc).

    Each DICOM control point carries its own jaw, MLC and gantry
    values from the matching Pinnacle control point (the final, appended
    control point repeats the last Pinnacle aperture).

    FinalCumulativeMetersetWeight equals the last control point's
    cumulative weight instead of a hardcoded "1".

    GantryAngle is written on every control point and
    GantryRotationDirection is derived per control point from the angle
    deltas, so beams that reverse direction mid-delivery are represented.
    """
    plan.logger.debug("Not using Step & Shoot")

    decimals = _collimation_decimals(machine)

    # Pinnacle's per-control-point Weight is the fraction of the beam's
    # meterset delivered in the segment *starting* at that point, so N
    # Pinnacle points normally describe N segments and need N+1 DICOM
    # control points -- a static beam stores one point of weight 1 and
    # Pinnacle's own export writes two.  An arc, however, stores a final
    # point of weight zero which is already the terminating point: adding
    # another duplicates the last aperture at an unchanged cumulative
    # weight and inflates NumberOfControlPoints by one (76 -> 77).
    trailing_zero_weight = False
    if len(metersetweight) > 2:
        try:
            trailing_zero_weight = float(metersetweight[-1]) == 0.0
        except (TypeError, ValueError):
            trailing_zero_weight = False


    total_cps = numctrlpts if trailing_zero_weight else numctrlpts + 1
    beam_ds.NumberOfControlPoints = total_cps
    plan.logger.debug(
        "Beam '%s': %d Pinnacle control point(s) -> %d DICOM control "
        "point(s) (the final Pinnacle weight is %szero).",
        beam["Name"],
        numctrlpts,
        total_cps,
        "" if trailing_zero_weight else "non-",
    )
    # SourceToSurfaceDistance (300A,0130) is defined on the Control Point
    # Sequence, not on the beam, so it is written per control point below.

    if numwedges > 0:
        beam_ds.WedgeSequence = _create_wedge_sequence(wedge_info)

    # --- Cumulative meterset weights --------
    cumulative_weights = []
    running = 0.0
    for j in range(total_cps):
        if j > 0:
            running += float(metersetweight[j])
        cumulative_weights.append(running)

    cumulative_weights, final_weight = _normalise_cumulative_weights(
        cumulative_weights, beam["Name"], plan.logger
    )
    beam_ds.FinalCumulativeMetersetWeight = _format_ds(final_weight)

    gantry_directions = _gantry_directions(
        cp_data_list, total_cps, gantryrotdir, beam["Name"], plan.logger
    )

    for j in range(total_cps):
        cp = _new_dataset()
        beam_ds.ControlPointSequence.append(cp)

        cp.ControlPointIndex = j
        cp.CumulativeMetersetWeight = _format_ds(cumulative_weights[j])

        # appended final CP repeats the last Pinnacle aperture.
        pinn_idx = min(j, len(cp_data_list) - 1)
        cp_entry = cp_data_list[pinn_idx]

        if j == 0:
            # First control point: all attributes must be present
            _populate_first_control_point(
                cp,
                beam_ds,
                beam,
                plan,
                beam_energy,
                doserate,
                gantry_directions[0],
                numwedges,
                cp_entry,
                iso_center,
                decimals,
            )
        else:
            # Subsequent control points carry only what changed since the
            # previous one, which is both what DICOM asks for and what
            # Pinnacle's own export does: a static beam's terminating
            # control point holds nothing but its cumulative weight.
            # Repeating unchanged collimation is legal but asserts a
            # machine movement that does not happen, and buries the
            # control points that do move.
            previous = cp_data_list[min(j - 1, len(cp_data_list) - 1)]

            if _collimation_changed(previous, cp_entry):
                cp.BeamLimitingDevicePositionSequence = (
                    _create_bld_position_entries(
                        cp_entry["x1"],
                        cp_entry["x2"],
                        cp_entry["y1"],
                        cp_entry["y2"],
                        cp_entry["leafpositions"],
                        decimals,
                    )
                )

            # The rotation direction accompanies the angle: it qualifies
            # the movement to the next control point, so writing one
            # without the other says nothing useful.  Direction reversals
            # mid-delivery are captured because the angle changes there too.
            if cp_entry["gantry"] != previous["gantry"]:
                cp.GantryAngle = cp_entry["gantry"]
                cp.GantryRotationDirection = gantry_directions[j]

            if cp_entry["collimator"] != previous["collimator"]:
                cp.BeamLimitingDeviceAngle = cp_entry["collimator"]
                cp.BeamLimitingDeviceRotationDirection = "NONE"

            if cp_entry["couch"] != previous["couch"]:
                cp.PatientSupportAngle = cp_entry["couch"]
                cp.PatientSupportRotationDirection = "NONE"

            _set_cp_ssd(cp, cp_entry)

    # Beam Limiting Device Sequence (beam level)
    _populate_beam_limiting_device_seq(
        beam_ds, p_count, plan.logger, machine_boundaries, machine
    )