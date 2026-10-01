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
import random
import re
import time

from pymedphys._dicom.create import set_default_transfer_syntax
from pymedphys._imports import pydicom
from pymedphys._pinnacle.pinnacle_exceptions import MissingCTImageError

from .constants import (
    GImplementationClassUID,
    GTransferSyntaxUID,
    RTSTRUCTModality,
    RTStructSOPClassUID,
    colors,
)
from .pinnacle_metadata import apply_approval_status, apply_equipment_stamps

# DICOM LO (Long String) maximum length — used for SeriesDescription.
_DICOM_LO_MAX = 64

# DICOM SH (Short String) maximum length — used for StructureSetLabel.
_DICOM_SH_MAX = 16


def _set_trial_series_description(ds, trial_name):
    """Append 'Struct: trial' to SeriesDescription, respecting the LO VR.

    IS: PLAN-08 — exported objects carry their trial name so that a
    reviewer can distinguish trials within a plan at the PACS/TPS end.
    """
    label = f"Struct: {trial_name}"
    base = (getattr(ds, "SeriesDescription", "") or "").strip()
    combined = f"{base} - {label}" if base else label
    if len(combined) > _DICOM_LO_MAX:
        combined = combined[: _DICOM_LO_MAX - 3].rstrip() + "..."
    ds.SeriesDescription = combined


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


# Canonical CT Image SOP Class UID
_CT_IMAGE_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.2"
# Study Component Management SOP Class (for referenced study)
_STUDY_COMPONENT_SOP_CLASS_UID = "1.2.840.10008.3.1.2.3.2"


def _find_closest_slice(image_info_list, z_coord_mm, patient_position="HFS"):
    """Find the image slice closest to a given z-coordinate.

    Parameters
    ----------
    image_info_list : list of dict
        Each dict must have 'TablePosition' and 'InstanceUID' keys.
    z_coord_mm : float
        The z-coordinate in mm (DICOM patient coordinates).
    patient_position : str
        Patient position string, e.g. 'HFS', 'FFS'.

    Returns
    -------
    pydicom.dataset.Dataset
        A ContourImageSequence item with ReferencedSOPClassUID and
        ReferencedSOPInstanceUID pointing to the closest slice.
    """
    closest_distance = float("inf")
    closest_uid = None

    for s in image_info_list:
        # Pinnacle stores table position in cm, z_coord is in mm (DICOM).
        # For head-first (HFS/HFP): DICOM z = -Pinnacle z  →  table_pos ≈ -z/10
        # For feet-first (FFS/FFP): DICOM z =  Pinnacle z  →  table_pos ≈  z/10
        if patient_position in ("FFS", "FFP"):
            pinnacle_z_cm = z_coord_mm / 10
        else:
            pinnacle_z_cm = -z_coord_mm / 10
        distance = abs(float(s["TablePosition"]) - pinnacle_z_cm)
        if distance < closest_distance:
            closest_distance = distance
            closest_uid = s["InstanceUID"]

    contour_image = _new_dataset()
    contour_image.ReferencedSOPClassUID = _CT_IMAGE_SOP_CLASS_UID
    if closest_uid is not None:
        contour_image.ReferencedSOPInstanceUID = closest_uid
    return contour_image


def _transform_point_for_position(
    curr_points, patient_position, coordinate_shift=(0.0, 0.0, 0.0)
):
    """Transform ROI contour points from Pinnacle coordinates to DICOM patient coords.

    Pinnacle stores coordinates in cm; DICOM uses mm. The sign conventions
    differ by patient orientation.

    Parameters
    ----------
    curr_points : list of str
        Three string values [x, y, z] from the plan.roi file.
    patient_position : str
        One of 'HFS', 'HFP', 'FFS', 'FFP'.
    coordinate_shift : tuple of float, optional
        ``(xshift, yshift, zshift)`` in mm for pre-v9 archives.
        Defaults to ``(0, 0, 0)`` (no shift).

    Returns
    -------
    list of float
        Transformed [x, y, z] in mm.
    """
    x, y, z = float(curr_points[0]), float(curr_points[1]), float(curr_points[2])

    transform_map = {
        "HFS": (x * 10, -y * 10, -z * 10),
        "HFP": (-x * 10, y * 10, -z * 10),
        "FFP": (x * 10, y * 10, z * 10),
        "FFS": (-x * 10, -y * 10, z * 10),
    }
    if patient_position not in transform_map:
        raise NotImplementedError(
            f"{patient_position} orientation is not supported for structure "
            f"coordinate transforms. Supported: {tuple(transform_map)}."
        )
    tx, ty, tz = transform_map[patient_position]
    return [
        tx + coordinate_shift[0],
        ty + coordinate_shift[1],
        tz + coordinate_shift[2],
    ]


# ---------------------------------------------------------------------------
# Isocenter detection
# ---------------------------------------------------------------------------


def find_iso_center(plan):
    """Determine the isocenter, CT centre and dose reference point for the plan.

    Searches through all points defined in the plan and uses a priority-based
    heuristic to identify the isocenter:
      1. Points with PoiInterpretedType containing 'ISO'
      2. Points named Iso / isocenter / isocentre / ISO
      3. CT Center / ct center / ct centre
      4. Points containing 'center' or 'iso' in the name (case-sensitive)
    When nothing matches, the isocenter is left unresolved (the previous
    arbitrary first-point fallback has been removed; the RTPLAN path raises
    IsocenterNotFoundError instead of guessing).
    """
    iso_center = []
    ct_center = []
    dose_ref_pt = []

    for point in plan.points:
        refpoint = plan.convert_point(point)

        # Check for isocenter by name (case-insensitive)
        name_l = point["Name"].lower()
        if any(tag in name_l for tag in ("iso", "isocenter", "isocentre")):
            iso_center = refpoint

        # Check for CT center (case-insensitive)
        if "ct center" in name_l or "ct centre" in name_l:
            ct_center = refpoint

        # Check for dose reference point (case-insensitive)
        if "drp" in name_l:
            dose_ref_pt = refpoint

        # Highest priority: explicit PoiInterpretedType (case-insensitive)
        if "PoiInterpretedType" in point:
            if "iso" in point["PoiInterpretedType"].lower():
                iso_center = refpoint
                plan.logger.debug("ISO Center located: %s", iso_center)

    # Fallback chain
    if len(iso_center) < 2:
        iso_center = ct_center
        plan.logger.debug("Isocenter not located, setting to CT center: %s", iso_center)

    if len(iso_center) < 2:
        plan.logger.debug(
            "Isocenter still not located, trying points with 'center' or 'iso' in name"
        )
        point_with_center = []
        point_with_iso = []

        for p in plan.points:
            pname_l = p["Name"].lower()
            if "center" in pname_l or "centre" in pname_l:
                point_with_center = p["refpoint"]
            elif "iso" in pname_l:
                point_with_iso = p["refpoint"]

        if len(point_with_center) > 1:
            iso_center = point_with_center
        elif len(point_with_iso) > 1:
            iso_center = point_with_iso
        else:
            # no isocenter-like point exists.  The isocenter now left unset
            # and the RTPLAN path fails explicitly (IsocenterNotFoundError)
            # rather than exporting a guessed geometry.
            plan.logger.warning(
                "No isocenter-like point could be identified among the plan "
                "points; the isocenter is left unresolved instead of "
                "defaulting to an arbitrary point."
            )

    plan.iso_center = iso_center
    plan.ct_center = ct_center
    plan.dose_ref_pt = dose_ref_pt
    plan.logger.debug("Isocenter: %s", iso_center)


# ---------------------------------------------------------------------------
# Points → DICOM
# ---------------------------------------------------------------------------


def read_points(ds, plan):
    """Read plan points (POIs) and add them to the DICOM dataset.

    Each point becomes:
      - An item in ROIContourSequence (with one POINT contour)
      - An item in StructureSetROISequence
      - An item in RTROIObservationsSequence
    """
    plan.roi_count = 0
    image_info = plan.primary_image.image_info
    image_header = plan.primary_image.image_header
    patient_position = image_header["patient_position"]
    frame_uid = image_info[0]["FrameUID"]

    for point in plan.points:
        plan.roi_count += 1
        refpoint = plan.convert_point(point)

        # --- ROI Contour ---
        roi_contour = _new_dataset()
        roi_contour.ReferencedROINumber = str(plan.roi_count)
        try:
            roi_contour.ROIDisplayColor = colors[point["Color"]]
        except KeyError:
            plan.logger.info(
                "POI color not known: %s — assigning a random color",
                point.get("Color"),
            )
            roi_contour.ROIDisplayColor = colors[random.choice(list(colors))]
        roi_contour.ContourSequence = _new_sequence()

        contour = _new_dataset()
        contour.ContourData = refpoint
        contour.ContourGeometricType = "POINT"
        contour.NumberOfContourPoints = 1
        contour.ContourImageSequence = _new_sequence()
        contour.ContourImageSequence.append(
            _find_closest_slice(image_info, refpoint[-1], patient_position)
        )
        roi_contour.ContourSequence.append(contour)
        ds.ROIContourSequence.append(roi_contour)

        # --- Structure Set ROI (separate dataset — not sharing with contour!) ---
        structure_set_roi = _new_dataset()
        structure_set_roi.ROINumber = plan.roi_count
        structure_set_roi.ROIName = point["Name"]
        structure_set_roi.ROIDescription = ""  # Type 3 but good practice to include
        structure_set_roi.ROIGenerationAlgorithm = "SEMIAUTOMATIC"
        structure_set_roi.ReferencedFrameOfReferenceUID = frame_uid
        ds.StructureSetROISequence.append(structure_set_roi)
        plan.logger.info("Exporting point: %s", point["Name"])

        # --- RT ROI Observations ---
        observation = _new_dataset()
        observation.ObservationNumber = plan.roi_count
        observation.ReferencedROINumber = plan.roi_count
        observation.RTROIInterpretedType = "MARKER"
        observation.ROIInterpreter = ""
        ds.RTROIObservationsSequence.append(observation)

    return ds


# ---------------------------------------------------------------------------
# ROI contours → DICOM (line-by-line parser for plan.roi)
# ---------------------------------------------------------------------------


def read_roi(ds, plan, skip_pattern):
    """Read ROI contours from the plan.roi file and add to the DICOM dataset.

    The plan.roi file uses a bespoke text format that is not proper YAML,
    so we parse it line by line. Each ROI becomes entries in
    ROIContourSequence, StructureSetROISequence, and RTROIObservationsSequence.
    """
    image_header = plan.primary_image.image_header
    image_info = plan.primary_image.image_info
    frame_uid = image_info[0]["FrameUID"]
    patient_position = image_header["patient_position"]
    coord_shift = plan.coordinate_shift  # (0,0,0) for v9+

    path_roi = os.path.join(plan.path, "plan.roi")
    plan.logger.debug("Will skip ROIs matching pattern[%s]", skip_pattern)
    plan.logger.debug("Reading ROI from: %s", path_roi)

    if not os.path.exists(path_roi):
        plan.logger.warning(
            "plan.roi not found at: %s — no ROI contours to export", path_roi
        )
        return ds

    # State variables for the line-by-line parser
    flag_skip_roi = False
    flag_points = False
    points = []
    first_points = []
    curvenum = 0
    roiinterpretedtype = "ORGAN"

    with open(path_roi) as f:
        for _, line in enumerate(f, 1):
            # ----- Skip mode: fast-forward through an excluded ROI -----
            if flag_skip_roi:
                if "}; // End of ROI" in line:
                    flag_skip_roi = False
                continue

            # ----- End of points for a curve -----
            if "};  // End of points for curve" in line:
                # Extract curve number from end-of-curve marker
                numfind = int(line.find("curve") + 5)
                curvenum = int(line[numfind:].strip())

                # Reference the current ROI contour and its curve
                roi_contour = ds.ROIContourSequence[plan.roi_count - 1]
                contour_item = roi_contour.ContourSequence[curvenum - 1]

                contour_item.NumberOfContourPoints = int(len(points) / 3)
                contour_item.ContourData = points

                # Link to the closest CT slice
                contour_item.ContourImageSequence = _new_sequence()
                if len(points) >= 3:
                    contour_item.ContourImageSequence.append(
                        _find_closest_slice(image_info, points[-1], patient_position)
                    )

                del points[:]
                flag_points = False
                continue

            # ----- Reading point data -----
            if flag_points:
                curr_points = line.split(" ")

                # Skip duplicate of the first point (closed contour)
                if curr_points == first_points:
                    continue

                if len(first_points) == 0:
                    first_points = curr_points

                transformed = _transform_point_for_position(
                    curr_points, patient_position, coord_shift
                )

                transformed = [round(v, 5) for v in transformed]

                points = points + transformed

            # ----- Beginning of a new ROI -----
            if "Beginning of ROI" in line:
                roi_name = line[22:].rstrip()
                plan.logger.debug("Start of ROI [%s]", roi_name)

                if re.match(skip_pattern, roi_name):
                    plan.logger.info("Skipping ROI [%s]", roi_name)
                    flag_skip_roi = True
                    continue

                plan.roi_count += 1

                # ROI Contour (separate dataset for the contour)
                roi_contour = _new_dataset()
                roi_contour.ReferencedROINumber = str(plan.roi_count)
                roi_contour.ContourSequence = _new_sequence()
                ds.ROIContourSequence.append(roi_contour)

                # Structure Set ROI (MUST be a separate dataset!)
                structure_set_roi = _new_dataset()
                structure_set_roi.ROINumber = plan.roi_count
                structure_set_roi.ROIName = roi_name
                structure_set_roi.ROIDescription = ""
                structure_set_roi.ROIGenerationAlgorithm = "SEMIAUTOMATIC"
                structure_set_roi.ReferencedFrameOfReferenceUID = frame_uid
                ds.StructureSetROISequence.append(structure_set_roi)

                # RT ROI Observations (placeholder, populated at end-of-ROI)
                observation = _new_dataset()
                ds.RTROIObservationsSequence.append(observation)

                roiinterpretedtype = "ORGAN"
                plan.logger.info("Exporting ROI: %s", roi_name)

            # ----- ROI interpreted type -----
            if "roiinterpretedtype:" in line:
                roiinterpretedtype = line.split(" ")[-1].replace("\n", "")

            # ----- ROI color -----
            if "color:" in line:
                roi_color = line.split(" ")[-1].replace("\n", "")
                roi_contour_item = ds.ROIContourSequence[plan.roi_count - 1]
                try:
                    roi_contour_item.ROIDisplayColor = colors[roi_color]
                except KeyError:
                    plan.logger.info("ROI Color not known: %s", roi_color)
                    new_color = random.choice(list(colors))
                    plan.logger.info("Instead, assigning color: %s", new_color)
                    roi_contour_item.ROIDisplayColor = colors[new_color]

            # ----- End of ROI -----
            if "}; // End of ROI" in line:
                observation = ds.RTROIObservationsSequence[plan.roi_count - 1]
                observation.ObservationNumber = plan.roi_count
                observation.ReferencedROINumber = plan.roi_count
                observation.RTROIInterpretedType = roiinterpretedtype
                observation.ROIInterpreter = ""

            # ----- ROI volume -----
            if "volume =" in line:
                vol = re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]
                ds.StructureSetROISequence[plan.roi_count - 1].ROIVolume = vol

            # ----- New curve within current ROI -----
            if "//  Curve " in line:
                first_points = []
                curvenum = re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]

                contour = _new_dataset()
                ds.ROIContourSequence[plan.roi_count - 1].ContourSequence.append(
                    contour
                )

            # ----- Number of points in current curve -----
            if "num_points =" in line:
                npts = re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]
                contour_item = ds.ROIContourSequence[
                    plan.roi_count - 1
                ].ContourSequence[int(curvenum) - 1]
                contour_item.ContourGeometricType = "CLOSED_PLANAR"
                contour_item.NumberOfContourPoints = npts

            # ----- Start reading point data -----
            if "points=" in line:
                flag_points = True

    plan.logger.debug("patient pos: %s", patient_position)
    return ds


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def convert_struct(plan, export_path, skip_pattern):
    """Export RTSTRUCT files for every trial in the plan.

    Although the structure geometry itself does not change between trials,
    each trial gets its own RTSTRUCT SOP Instance UID so that the matching
    RTPLAN/RTDOSE produced for that trial reference an RTSTRUCT they can
    actually find. Sharing one struct UID across trials would force every
    trial's RP/RD to point at the same RS, which a number of PACS / TPS
    treat as duplicate-and-drop.
    """
    if not plan.primary_image:
        plan.logger.error(
            "No primary image found for plan. Unable to generate RTSTRUCT."
        )
        raise MissingCTImageError("Plan has no primary image associated with it.")

    for trial_info in plan.trials:
        plan.active_trial = trial_info["Name"]
        plan.logger.info("Exporting RTSTRUCT for trial: %s", trial_info["Name"])

        uids = plan.generate_uids_for_trial(trial_info)
        convert_struct_for_trial(
            plan,
            trial_info,
            struct_instance_uid=uids["struct"],
            series_instance_uid=uids["series_struct"],
            export_path=export_path,
            skip_pattern=skip_pattern,
        )


def convert_struct_for_trial(
    plan,
    trial_info,
    struct_instance_uid,
    series_instance_uid,
    export_path,
    skip_pattern,
):
    """Write a single RTSTRUCT DICOM file for one specific trial."""

    patient_info = plan.pinnacle.patient_info
    plan_info = plan.plan_info
    image_info = plan.primary_image.image_info
    first_image = image_info[0]

    # --- File meta ---
    file_meta = _new_dataset()
    file_meta.MediaStorageSOPClassUID = RTStructSOPClassUID
    file_meta.TransferSyntaxUID = GTransferSyntaxUID
    file_meta.MediaStorageSOPInstanceUID = struct_instance_uid
    file_meta.ImplementationClassUID = GImplementationClassUID

    safe_trial = _sanitize_for_filename(trial_info.get("Name"))
    struct_filename = f"RS.{safe_trial}.{struct_instance_uid}.dcm"

    ds = pydicom.dataset.FileDataset(
        struct_filename, {}, file_meta=file_meta, preamble=b"\x00" * 128
    )

    struct_series_uid = series_instance_uid

    # --- Character set and timestamps ---
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.InstanceCreationDate = time.strftime("%Y%m%d")
    ds.InstanceCreationTime = time.strftime("%H%M%S")

    # --- SOP / Modality ---
    ds.SOPClassUID = RTStructSOPClassUID
    ds.SOPInstanceUID = struct_instance_uid
    ds.Modality = RTSTRUCTModality
    ds.AccessionNumber = ""
    ds.Manufacturer = ""  # Type 2; overwritten by apply_equipment_stamps

    # StationName (Type 3): left empty by default; populated from
    # DICOM_EQUIPMENT.STATION_NAME by apply_equipment_stamps when
    # configured
    ds.StationName = ""
    ds.ManufacturerModelName = plan_info.get("ToolType", "")
    # Wrapped in a list for consistent VM with rtdose.py / rtplan.py.
    ds.SoftwareVersions = [plan_info["PinnacleVersionDescription"]]

    # Apply site-specific equipment identification stamps from config
    apply_equipment_stamps(
        ds,
        plan.pinnacle.equipment_cfg,
        pinnacle_model=plan_info.get("ToolType", ""),
        pinnacle_sw=plan_info.get("PinnacleVersionDescription", ""),
    )

    # --- Referenced Study ---
    ds.ReferencedStudySequence = _new_sequence()
    ref_study = _new_dataset()
    ref_study.ReferencedSOPClassUID = _STUDY_COMPONENT_SOP_CLASS_UID
    ref_study.ReferencedSOPInstanceUID = first_image["StudyInstanceUID"]
    ds.ReferencedStudySequence.append(ref_study)

    # --- Study / Series ---
    ds.StudyInstanceUID = first_image["StudyInstanceUID"]
    ds.SeriesInstanceUID = struct_series_uid
    ds.SeriesNumber = "1"
    ds.StudyID = plan.primary_image.image["StudyID"]

    # carry the trial name so reviewers can distinguish trials
    # within a plan at the PACS/TPS end (LO VR, truncated to 64).
    _set_trial_series_description(ds, trial_info.get("Name", ""))

    # --- Patient ---
    ds.PatientID = patient_info["MedicalRecordNumber"]
    ds.PatientName = patient_info["FullName"]
    ds.PatientSex = patient_info.get("Gender", "")[:1]
    ds.PatientBirthDate = patient_info["DOB"]
    ds.ReferringPhysicianName = patient_info["ReferringPhysician"]
    ds.PhysiciansOfRecord = patient_info["RadiationOncologist"]
    ds.StudyDescription = patient_info["Comment"]

    # --- Structure Set identification ---
    # StructureSetLabel is VR SH (max 16 chars).  Pinnacle plan names are
    # frequently longer (e.g. "CopyOf_1_LtBreast" = 17), so it is clipped
    # here; the full name remains in SeriesDescription (LO).
    _label = str(plan_info["PlanName"] or "")
    if len(_label) > _DICOM_SH_MAX:
        plan.logger.warning(
            "StructureSetLabel value %r is %d chars, exceeding the DICOM SH "
            "limit of %d; truncated to %r (full name retained in "
            "SeriesDescription).",
            _label,
            len(_label),
            _DICOM_SH_MAX,
            _label[:_DICOM_SH_MAX],
        )
        _label = _label[:_DICOM_SH_MAX]
    ds.StructureSetLabel = _label
    ds.StructureSetName = "POIandROI"
    ds.StructureSetDescription = ""  # Type 3 — include empty for completeness

    # Dates from trial or plan
    datetimesplit = plan_info["ObjectVersion"]["WriteTimeStamp"].split()
    trial_info_for_dates = plan.trial_info
    if trial_info_for_dates:
        datetimesplit = trial_info_for_dates["ObjectVersion"]["WriteTimeStamp"].split()

    study_date = datetimesplit[0].replace("-", "")
    study_time = datetimesplit[1].replace(":", "")
    ds.StructureSetDate = study_date
    ds.StructureSetTime = study_time
    ds.StudyDate = study_date
    ds.StudyTime = study_time

    # --- Referenced Frame of Reference ---
    # Build the full chain: FrameOfReference → RTReferencedStudy → RTReferencedSeries
    ds.ReferencedFrameOfReferenceSequence = _new_sequence()
    frame_ref = _new_dataset()
    frame_ref.FrameOfReferenceUID = first_image["FrameUID"]
    ds.ReferencedFrameOfReferenceSequence.append(frame_ref)

    # RT Referenced Study
    frame_ref.RTReferencedStudySequence = _new_sequence()
    rt_ref_study = _new_dataset()
    rt_ref_study.ReferencedSOPClassUID = _STUDY_COMPONENT_SOP_CLASS_UID
    rt_ref_study.ReferencedSOPInstanceUID = first_image["StudyInstanceUID"]
    frame_ref.RTReferencedStudySequence.append(rt_ref_study)

    # RT Referenced Series
    rt_ref_study.RTReferencedSeriesSequence = _new_sequence()
    rt_ref_series = _new_dataset()
    rt_ref_series.SeriesInstanceUID = first_image["SeriesUID"]
    rt_ref_study.RTReferencedSeriesSequence.append(rt_ref_series)

    # Contour images for every slice
    rt_ref_series.ContourImageSequence = _new_sequence()
    for info in image_info:
        contour_image = _new_dataset()
        contour_image.ReferencedSOPClassUID = _CT_IMAGE_SOP_CLASS_UID
        contour_image.ReferencedSOPInstanceUID = info["InstanceUID"]
        rt_ref_series.ContourImageSequence.append(contour_image)

    # --- Main structure sequences ---
    ds.ROIContourSequence = _new_sequence()
    ds.StructureSetROISequence = _new_sequence()
    ds.RTROIObservationsSequence = _new_sequence()

    # --- Populate structures ---
    find_iso_center(plan)
    ds = read_points(ds, plan)
    ds = read_roi(ds, plan, skip_pattern)

    # Derived from Pinnacle PlanLockStatus (locked → APPROVED
    # with reviewer/timestamp audit fields, unlocked → UNAPPROVED).
    apply_approval_status(ds, plan)

    # Set the transfer syntax
    set_default_transfer_syntax(ds)

    # --- Save ---
    output_file = os.path.join(export_path, struct_filename)
    plan.logger.info("Creating Struct file: %s", output_file)
    ds.save_as(output_file, enforce_file_format=True)
