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


import math
import os
import re
import struct
import time

from pymedphys._dicom.orientation import IMAGE_ORIENTATION_MAP
from pymedphys._imports import numpy as np
from pymedphys._imports import pydicom

# DICOM DS (Decimal String) maximum length — PS3.5 Table 6.2-1.
_DICOM_DS_MAX = 16


def _format_ds_value(value):
    """Render a float as a DICOM DS string of at most 16 characters.

    DoseGridScaling is VR DS, so ``str(float)`` (which can be 21+ chars,
    e.g. "0.0004231092002428742") is non-conformant and is rejected or
    silently truncated by strict readers.  The most precise representation
    that fits is used; DS permits E-notation, so tiny/huge magnitudes stay
    representable.  Relative precision retained is ~1e-11, which is far
    below any dosimetric significance.
    """
    value = float(value)
    for precision in range(15, 0, -1):
        text = f"{value:.{precision}g}"
        if len(text) <= _DICOM_DS_MAX:
            return text
    # Unreachable for finite floats, but never emit an over-long value.
    return f"{value:.6e}"[:_DICOM_DS_MAX]


from pymedphys._pinnacle.pinnacle_exceptions import (
    InvalidDoseNormalizationError,
    MissingBeamDoseError,
    MissingCTImageError,
    MissingTrialBeamsError,
)

from .constants import (
    GImplementationClassUID,
    GTransferSyntaxUID,
    RTDOSEModality,
    RTDoseSOPClassUID,
    RTPlanSOPClassUID,
)
from .pinnacle_metadata import apply_equipment_stamps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_dataset():
    """Shorthand for creating a new empty DICOM Dataset."""
    return pydicom.dataset.Dataset()


def _new_sequence():
    """Shorthand for creating a new empty DICOM Sequence."""
    return pydicom.sequence.Sequence()


# Sign maps for dose origin calculation by patient position.
# Keys: patient_position → (x_sign, y_sign, z_sign)
_DOSE_ORIGIN_SIGNS = {
    "HFS": (1, -1, -1),
    "HFP": (-1, 1, -1),
    "FFS": (-1, -1, 1),
    "FFP": (1, 1, 1),
}

# Image position patient shift directions.
# Keys: patient_position → (y_shift_sign, z_shift_sign)
# y_shift_sign: +1 means add ydoseshift, -1 means subtract
# z_shift_sign: applied to the z shift from origin to grid edge.
#
# For head-first orientations the dose origin (after sign conversion)
# lands at the *far* z-edge of the grid, so we shift by -z_shift to
# reach the IPP (near edge where GFOV = 0).
# For feet-first orientations the sign conversion already places the
# origin at the IPP edge, so no z-shift is needed.
_IMAGE_POSITION_SHIFTS = {
    "HFS": (-1, -1),
    "HFP": (+1, -1),
    "FFS": (-1, 0),
    "FFP": (+1, 0),
}

_SUPPORTED_ORIENTATIONS = ("HFS", "HFP", "FFS", "FFP")


def construct_dose_from_binary(binary_data, array):
    """Read binary data into an empty dose array.

    The binary format is big-endian 32-bit floats, ordered Z (high-to-low),
    Y (low-to-high), X (low-to-high).
    """
    X, Y, Z = array.shape
    idx = 0
    for z in range(Z - 1, -1, -1):
        for y in range(Y):
            for x in range(X):
                data_element = binary_data[idx : idx + 4]
                value = struct.unpack(">f", data_element)[0]
                array[x, y, z] = value
                idx += 4
    return array


def read_binary_data(binary_file):
    """Check if the supplied binary file exists and is non-empty.

    Returns the binary data if valid, or False if the file is missing or
    contains only zeros.
    """
    if os.path.isfile(binary_file):
        size = os.path.getsize(binary_file)
        with open(binary_file, "rb") as b:
            data = b.read()
            if all(byte == 0 for byte in data):
                return False
            return data
    return False


def trilinear_interpolation(idx, grid):
    """Return trilinear interpolated value for a voxel at fractional index.

    Parameters
    ----------
    idx : list of float
        Fractional [x, y, z] indices into the grid.
    grid : numpy.ndarray
        3D dose grid.

    Returns
    -------
    float
        Interpolated value.
    """
    # Clamp the integer indices so the 8-corner sample never reads outside
    # the grid.  For an axis of size N the lower corner must lie in
    # [0, N-2] so that the +1 upper corner stays within [0, N-1].  A
    # prescription point on or beyond the grid edge therefore degrades
    # gracefully to the nearest in-grid voxel instead of raising IndexError.
    shape = grid.shape
    int_idx = []
    frac_idx = []
    for d, f in enumerate(idx):
        lo = int(math.floor(f))
        max_lo = max(shape[d] - 2, 0)
        clamped = min(max(lo, 0), max_lo)
        int_idx.append(clamped)
        # Preserve the fractional offset only while inside the grid.
        frac_idx.append(min(max(f - clamped, 0.0), 1.0))

    # Sample the 8 corner values of the enclosing voxel (upper corner index
    # additionally clamped to the last valid voxel on every axis).
    x_max, y_max, z_max = shape[0] - 1, shape[1] - 1, shape[2] - 1
    corners = [[[0.0] * 2 for _ in range(2)] for _ in range(2)]
    for x in range(2):
        for y in range(2):
            for z in range(2):
                corners[x][y][z] = grid[
                    min(int_idx[0] + x, x_max),
                    min(int_idx[1] + y, y_max),
                    min(int_idx[2] + z, z_max),
                ]

    # Interpolate along X
    interp_x = [[0.0] * 2 for _ in range(2)]
    for y in range(2):
        for z in range(2):
            interp_x[y][z] = (
                corners[0][y][z] * (1 - frac_idx[0]) + corners[1][y][z] * frac_idx[0]
            )

    # Interpolate along Y
    interp_xy = [0.0, 0.0]
    for z in range(2):
        interp_xy[z] = interp_x[0][z] * (1 - frac_idx[1]) + interp_x[1][z] * frac_idx[1]

    # Interpolate along Z
    return interp_xy[0] * (1 - frac_idx[2]) + interp_xy[1] * frac_idx[2]


def _get_dose_grid_value(trial_info, axis, property_name):
    """Get a dose grid property from trial info.

    Example: _get_dose_grid_value(trial_info, 'X', 'Dimension') reads
    trial_info["DoseGrid .Dimension .X"].
    """
    return trial_info[f"DoseGrid .{property_name} .{axis}"]


def _get_dose_dimensions(trial_info):
    """Return (dim_x, dim_y, dim_z) for the dose grid."""
    return (
        int(_get_dose_grid_value(trial_info, "X", "Dimension")),
        int(_get_dose_grid_value(trial_info, "Y", "Dimension")),
        int(_get_dose_grid_value(trial_info, "Z", "Dimension")),
    )


def _get_voxel_sizes_mm(trial_info):
    """Return (vx, vy, vz) voxel sizes in mm (Pinnacle stores cm)."""
    return (
        _get_dose_grid_value(trial_info, "X", "VoxelSize") * 10,
        _get_dose_grid_value(trial_info, "Y", "VoxelSize") * 10,
        _get_dose_grid_value(trial_info, "Z", "VoxelSize") * 10,
    )


def _compute_image_position_patient(
    dose_origin, voxel_mm, dimensions, patient_position
):
    """Compute ImagePositionPatient for the dose grid.

    The dose origin from Pinnacle, after sign conversion, sits at a
    specific corner of the dose grid.  DICOM needs the corner where
    GridFrameOffsetVector = 0 (the ImagePositionPatient).

    For head-first orientations the sign conversion places the origin at
    the far z-edge, so we shift by -(dim-1)*vz to reach the IPP corner.
    For feet-first orientations the origin is already at the IPP corner,
    so no z-shift is applied.
    """
    vx, vy, vz = voxel_mm
    dim_x, dim_y, dim_z = dimensions
    ox, oy, oz = dose_origin

    y_shift = vy * dim_y - vy
    z_shift = vz * dim_z - vz

    y_sign, z_sign = _IMAGE_POSITION_SHIFTS[patient_position]

    return [
        ox,
        oy + y_sign * y_shift,
        oz + z_sign * z_shift,
    ]


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def convert_dose(plan, export_path):
    """Export RTDose files for all trials in the plan.

    For each trial, generates unique UIDs and creates an RTDOSE DICOM file.
    """
    if not plan.primary_image:
        plan.logger.error("No primary image found for plan. Unable to generate RTDOSE.")
        raise MissingCTImageError("Plan has no primary image associated with it.")

    patient_info = plan.pinnacle.patient_info
    plan_info = plan.plan_info
    image_info = plan.primary_image.image_info[0]
    patient_position = plan.patient_position

    if patient_position not in _SUPPORTED_ORIENTATIONS:
        raise NotImplementedError(
            f"{patient_position} orientation not supported. "
            f"Only: {_SUPPORTED_ORIENTATIONS}"
        )

    # --- Build base DICOM dataset (plan-level metadata, shared across trials) ---
    file_meta = _new_dataset()
    file_meta.MediaStorageSOPClassUID = RTDoseSOPClassUID
    file_meta.TransferSyntaxUID = GTransferSyntaxUID
    file_meta.ImplementationClassUID = GImplementationClassUID

    ds = pydicom.dataset.FileDataset(
        "RD.dcm", {}, file_meta=file_meta, preamble=b"\x00" * 128
    )

    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.InstanceCreationDate = time.strftime("%Y%m%d")
    ds.InstanceCreationTime = time.strftime("%H%M%S")
    ds.SOPClassUID = RTDoseSOPClassUID

    ds.AccessionNumber = ""
    ds.Modality = RTDOSEModality
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
    ds.FrameOfReferenceUID = image_info["FrameUID"]
    ds.StudyID = plan.primary_image.image["StudyID"]

    ds.ImageOrientationPatient = IMAGE_ORIENTATION_MAP[patient_position]
    ds.PositionReferenceIndicator = ""

    # Pixel encoding
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0

    # Dose type
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = "PLAN"

    # TissueHeterogeneityCorrection: set per-trial below based on whether
    # the trial uses ROI density overrides.  Default to IMAGE.
    ds.TissueHeterogeneityCorrection = "IMAGE"

    # Referenced RT Plan (placeholder — updated per trial)
    ds.ReferencedRTPlanSequence = _new_sequence()
    ref_plan = _new_dataset()
    ref_plan.ReferencedSOPClassUID = RTPlanSOPClassUID
    ds.ReferencedRTPlanSequence.append(ref_plan)

    # --- Dose origin sign convention ---
    x_sign, y_sign, z_sign = _DOSE_ORIGIN_SIGNS[patient_position]
    coord_shift = plan.coordinate_shift  # (0,0,0) for v9+

    # --- Process each trial ---
    for trial_info in plan.trials:
        plan.active_trial = trial_info["Name"]
        plan.logger.info("Exporting Dose for trial: %s", trial_info["Name"])

        uids = plan.generate_uids_for_trial(trial_info)
        dose_uid = uids["dose"]
        plan_uid = uids["plan"]
        series_uid = uids["series_dose"]

        dose_origin = [
            x_sign * _get_dose_grid_value(trial_info, "X", "Origin") * 10
            + coord_shift[0],
            y_sign * _get_dose_grid_value(trial_info, "Y", "Origin") * 10
            + coord_shift[1],
            z_sign * _get_dose_grid_value(trial_info, "Z", "Origin") * 10
            + coord_shift[2],
        ]

        # Determine TissueHeterogeneityCorrection for this trial.
        # If the trial has any ROI density overrides, use IMAGE\ROI_OVERRIDE.
        roi_overrides = trial_info.get("ROIOverrideList", None)
        has_roi_override = False
        if roi_overrides:
            # ROIOverrideList may be a list of overrides; check if any
            # have a non-default density override set.
            if isinstance(roi_overrides, list):
                has_roi_override = len(roi_overrides) > 0
            elif isinstance(roi_overrides, dict):
                has_roi_override = True

        if has_roi_override:
            ds.TissueHeterogeneityCorrection = ["IMAGE", "ROI_OVERRIDE"]
        else:
            ds.TissueHeterogeneityCorrection = "IMAGE"

        try:
            _convert_dose_for_trial(
                plan,
                trial_info,
                dose_uid,
                plan_uid,
                series_uid,
                dose_origin,
                patient_position,
                ds,
                export_path,
            )
        except (MissingTrialBeamsError, MissingBeamDoseError) as exc:
            plan.logger.warning(
                "Skipping RTDOSE for trial '%s': %s", trial_info["Name"], exc
            )
            continue
        except InvalidDoseNormalizationError as exc:
            # Failed on purpose — surface loudly so the operator
            # knows this trial's dose export was refused, not merely absent.
            plan.logger.error(
                "RTDOSE export failed for trial '%s': %s", trial_info["Name"], exc
            )
            continue


# ---------------------------------------------------------------------------
# Per-trial dose conversion
# ---------------------------------------------------------------------------


def _convert_dose_for_trial(
    plan,
    trial_info,
    dose_uid,
    plan_uid,
    series_uid,
    dose_origin,
    patient_position,
    ds,
    export_path,
):
    """Convert dose for a specific trial and save the RTDOSE DICOM file."""

    trial_name = trial_info.get("Name", "Unknown")
    dim_x, dim_y, dim_z = _get_dose_dimensions(trial_info)
    vx, vy, vz = _get_voxel_sizes_mm(trial_info)

    # --- Compute ImagePositionPatient ---
    image_position_patient = _compute_image_position_patient(
        dose_origin,
        (vx, vy, vz),
        (dim_x, dim_y, dim_z),
        patient_position,
    )

    # --- Update trial-specific DICOM fields ---
    ds.SOPInstanceUID = dose_uid
    ds.file_meta.MediaStorageSOPInstanceUID = dose_uid
    ds.SeriesInstanceUID = series_uid

    # Study date/time (DICOM compliance — must be present)
    datetimesplit = plan.plan_info["ObjectVersion"]["WriteTimeStamp"].split()
    if trial_info and "ObjectVersion" in trial_info:
        datetimesplit = trial_info["ObjectVersion"]["WriteTimeStamp"].split()
    ds.StudyDate = datetimesplit[0].replace("-", "")
    ds.StudyTime = datetimesplit[1].replace(":", "")

    ds.SeriesDescription = f"Dose: {trial_name}"
    ds.InstanceNumber = "1"

    rd_filename = f"RD.{trial_name}.{dose_uid}.dcm"
    ds.filename = rd_filename

    ds.ImagePositionPatient = image_position_patient
    ds.NumberOfFrames = dim_z
    ds.Rows = dim_y
    ds.Columns = dim_x
    ds.PixelSpacing = [vx, vy]
    ds.SliceThickness = vz

    ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = plan_uid

    # Grid frame offset vector — direction depends on orientation.
    # For head-first the frame normal is +z, so offsets are positive.
    # For feet-first the frame normal is -z, so offsets are negative
    # (frames still march from lower to higher z in patient coords).
    if patient_position in ("FFS", "FFP"):
        ds.GridFrameOffsetVector = [-p * vz for p in range(dim_z)]
    else:
        ds.GridFrameOffsetVector = [p * vz for p in range(dim_z)]

    # --- Sum beam doses ---
    summed_pixel_values = _sum_beam_doses(
        plan,
        trial_info,
        ds,
        patient_position,
        dim_x,
        dim_y,
        dim_z,
        (vx, vy, vz),
        image_position_patient,
    )

    # --- Scale and encode pixel data ---
    if not summed_pixel_values:
        plan.logger.error(
            "No usable beam dose was accumulated for trial '%s'; unable to "
            "generate RTDOSE.",
            trial_name,
        )
        raise MissingBeamDoseError(
            f"No usable beam dose accumulated for trial: {trial_name}"
        )
    # Map the maximum dose to the full unsigned 16-bit range (65535),
    # matching Pinnacle's own export, instead of the previous
    # 16384 divisor which quantised the grid to ~14 effective bits.
    scale = max(summed_pixel_values) / 65535
    # DoseGridScaling is VR DS (max 16 chars) — a bare float repr overflows
    # it and produces a non-conformant value.
    ds.DoseGridScaling = _format_ds_value(scale)
    plan.logger.debug(
        "Dose Grid Scaling: %s (DS-encoded: %s)", scale, ds.DoseGridScaling
    )

    if scale != 0:
        # Clamp to the unsigned 16-bit range so the encoded values agree with
        # the declared PixelRepresentation (0 = unsigned).  Dose is
        # non-negative, so the lower clamp is purely defensive against
        # floating-point rounding.
        pixel_values = [
            min(max(int(round(v / scale)), 0), 65535) for v in summed_pixel_values
        ]
    else:
        plan.logger.warning(
            "All summed dose values are zero for trial '%s'; DoseGridScaling "
            "is zero and the resulting RTDOSE will be entirely zero. Verify "
            "this is a genuinely empty dose and not a load/scaling failure.",
            trial_name,
        )
        pixel_values = [0] * len(summed_pixel_values)

    # --- Reverse frame order for feet-first orientations ---
    # _sum_beam_doses always outputs frames in ascending Pinnacle z-index
    # order (z=0 first = lowest Pinnacle z).  For HFS, Pinnacle's own RTDOSE
    # exporter uses this same ordering, so viewers see correct results.
    # For FFS/FFP, Pinnacle reverses the frame order so that frame 0
    # corresponds to the IPP position (highest DICOM z, where GFOV=0).
    # Without this reversal the dose volume appears z-flipped.
    if patient_position in ("FFS", "FFP"):
        pixels_per_frame = dim_x * dim_y
        frames = [
            pixel_values[i * pixels_per_frame : (i + 1) * pixels_per_frame]
            for i in range(dim_z)
        ]
        frames.reverse()
        pixel_values = [v for frame in frames for v in frame]

    # Pack as little-endian unsigned 16-bit to agree with the declared
    # PixelRepresentation = 0 (unsigned) and the Explicit VR LE transfer syntax.
    ds.PixelData = struct.pack("<%dH" % len(pixel_values), *pixel_values)

    ds.FrameIncrementPointer = ds.data_element("GridFrameOffsetVector").tag

    # --- Save ---
    output_file = os.path.join(export_path, rd_filename)
    plan.logger.info("Creating Dose file: %s", output_file)
    ds.save_as(output_file, enforce_file_format=True)


def _sum_beam_doses(
    plan,
    trial_info,
    ds,
    patient_position,
    dim_x,
    dim_y,
    dim_z,
    voxel_mm,
    image_position_patient,
):
    """Sum the dose contributions from all beams in a trial.

    Returns the summed pixel values list, already in the correct frame order
    for DICOM.
    """
    vx, vy, vz = voxel_mm
    origin = list(image_position_patient)
    trial_name = trial_info.get("Name", "Unknown")

    beam_list = trial_info["BeamList"] if trial_info["BeamList"] else []
    if not beam_list:
        plan.logger.warning(
            "No Beams found in Trial: %s. Unable to generate RTDOSE.", trial_name
        )
        raise MissingTrialBeamsError(f"No Beams found in Trial: {trial_name}")

    summed_pixel_values = []
    empty_beam_count = 0

    for beam in beam_list:
        plan.logger.info("Exporting Dose for beam: %s", beam["Name"])

        # --- Locate and validate binary dose file ---
        binary_id = re.findall("\\d+", beam["DoseVolume"])[0]
        filled_binary_id = str(binary_id).zfill(3)
        binary_file = os.path.join(plan.path, f"plan.Trial.binary.{filled_binary_id}")

        binary_data = read_binary_data(binary_file)
        if binary_data is False:
            plan.logger.warning(
                "No Dose found for beam: %s. Skipping beam.", beam["Name"]
            )
            empty_beam_count += 1
            if empty_beam_count == len(beam_list):
                plan.logger.error(
                    "All beams in plan are missing dose. Unable to generate RTDOSE."
                )
                raise MissingBeamDoseError("All beams in plan are missing dose.")
            continue

        # Validate the binary size against the declared dose grid before
        # unpacking, so a truncated file fails cleanly instead of raising a
        # struct.error mid-read, and an over-long file is flagged.
        expected_bytes = dim_x * dim_y * dim_z * 4
        if len(binary_data) < expected_bytes:
            plan.logger.error(
                "Dose binary for beam '%s' is %d bytes but the dose grid "
                "(%dx%dx%d) requires %d. Skipping beam to avoid reading past "
                "the buffer.",
                beam["Name"],
                len(binary_data),
                dim_x,
                dim_y,
                dim_z,
                expected_bytes,
            )
            empty_beam_count += 1
            if empty_beam_count == len(beam_list):
                raise MissingBeamDoseError(
                    "All beams in plan have missing or undersized dose."
                )
            continue
        if len(binary_data) > expected_bytes:
            plan.logger.warning(
                "Dose binary for beam '%s' is larger than the declared dose "
                "grid (%d > %d bytes); extra trailing bytes will be ignored.",
                beam["Name"],
                len(binary_data),
                expected_bytes,
            )

        # --- Prescription and scaling ---
        prescription = [
            p
            for p in trial_info["PrescriptionList"]
            if p["Name"] == beam["PrescriptionName"]
        ][0]

        # Find the prescription point
        plan.logger.debug("PrescriptionPointName: %s", beam["PrescriptionPointName"])
        prescription_point = []
        for p in plan.points:
            if p["Name"] == beam["PrescriptionPointName"]:
                plan.logger.debug(
                    "Presc Point: %s %s %s %s",
                    p["Name"],
                    p["XCoord"],
                    p["YCoord"],
                    p["ZCoord"],
                )
                prescription_point = plan.convert_point(p)
                break

        if len(prescription_point) < 3:
            plan.logger.warning(
                "No valid prescription point found for beam! Beam will be ignored "
                "for Dose conversion. Dose will most likely be incorrect"
            )
            continue

        plan.logger.debug("Presc Point Dicom: %s, %s", p["Name"], prescription_point)

        num_fractions = prescription["NumberOfFractions"]
        total_prescription = beam["MonitorUnitInfo"]["PrescriptionDose"] * num_fractions
        plan.logger.debug("Total Prescription %s", total_prescription)

        # --- Read dose grid and compute beam MU ---
        """Each per-beam binary file stores the dose contribution from that beam expressed as dose per unit weight
        — specifically, dose normalised to the beam's dose weight (also called beam weight or field weight), not dose per raw MU.
        Pinnacle internally works with a dose weight system rather than directly with MUs during optimisation and storage.
        """
        dose_per_unit_weight_grid = np.zeros((dim_x, dim_y, dim_z))
        dose_per_unit_weight_grid = construct_dose_from_binary(
            binary_data, dose_per_unit_weight_grid
        )

        spacing = [vx, vy, vz]

        # Get the fractional index of the prescription point within the grid.
        #
        # construct_dose_from_binary reverses the z-axis (binary high-to-low
        # is mapped so that array z=Z-1 = first binary = Pinnacle origin,
        # array z=0 = last binary).  The net effect is:
        #
        #   HFS/HFP: dose_per_unit_weight_grid[0,0,0] is at IPP_z → idx_z = 0 at IPP ✓
        #   FFS/FFP: dose_per_unit_weight_grid[0,0,0] is at IPP_z + (Z-1)*vz (the far end
        #            from IPP) → idx_z needs a (Z-1) offset.
        #
        # The orientation_matrix handles x and y correctly for all
        # orientations; only the z-axis needs the feet-first correction.
        orientation_matrix = np.zeros((3, 3))
        orientation_matrix[0, :] = IMAGE_ORIENTATION_MAP[patient_position][:3]
        orientation_matrix[1, :] = IMAGE_ORIENTATION_MAP[patient_position][3:]
        orientation_matrix[2, :] = np.cross(
            orientation_matrix[0, :], orientation_matrix[1, :]
        )

        idx = [0.0, 0.0, 0.0]
        for i in range(3):
            idx[i] = -(origin[i] - prescription_point[i]) / spacing[i]
            idx[i] *= orientation_matrix[i, i]

        if patient_position in ("FFS", "FFP"):
            idx[2] += dim_z - 1

        plan.logger.debug("Index of prescription point within grid: %s", idx)

        dims = (dim_x, dim_y, dim_z)
        if any(idx[i] < 0 or idx[i] > dims[i] - 1 for i in range(3)):
            plan.logger.warning(
                "Prescription point for beam '%s' falls outside the dose grid "
                "(index %s, grid %s); the nearest in-grid voxel will be used "
                "for the monitor-unit calculation.",
                beam["Name"],
                idx,
                dims,
            )

        cGy_per_unit_weight = trilinear_interpolation(idx, dose_per_unit_weight_grid)
        plan.logger.debug("cGy_per_unit_weight: %s", cGy_per_unit_weight)

        if cGy_per_unit_weight != 0:
            beam_weight = (total_prescription / cGy_per_unit_weight) / num_fractions
        else:
            # A beam that carries a non-zero prescription dose but
            # interpolates to zero at its prescription point cannot have its
            # monitor units derived — including it with zero weight would
            # silently under-report the PLAN summation, so fail this trial's
            # dose export.  A genuinely zero-prescription beam (e.g. a setup
            # field) legitimately contributes nothing and conversion carries
            # on with a warning.
            prescription_dose = beam["MonitorUnitInfo"]["PrescriptionDose"]
            if prescription_dose:
                raise InvalidDoseNormalizationError(
                    f"Beam '{beam['Name']}' has a non-zero prescription dose "
                    f"({prescription_dose} cGy/fraction) but the interpolated "
                    f"dose at its prescription point "
                    f"'{beam['PrescriptionPointName']}' is zero, so its "
                    f"monitor units cannot be derived. The PLAN dose "
                    f"summation would under-report total dose; verify the "
                    f"prescription point lies within the beam's dose region."
                )
            beam_weight = 0
            plan.logger.warning(
                "Interpolated dose at the prescription point for beam '%s' is "
                "zero and its prescription dose is also zero; this beam will "
                "contribute no dose to the PLAN summation.",
                beam["Name"],
            )
        plan.logger.debug("Beam MU: %s", beam_weight)

        # --- Convert dose grid to pixel values ---
        pixel_data_list = []
        for z in range(dim_z - 1, -1, -1):
            for y in range(dim_y):
                for x in range(dim_x):
                    value = (
                        num_fractions
                        * dose_per_unit_weight_grid[x, y, z]
                        * beam_weight
                        / 100
                    )
                    pixel_data_list.append(value)

        # Reorder into DICOM frame order
        main_pix_array = []
        pixels_per_frame = dim_x * dim_y
        for h in range(dim_z):
            frame_start = h * pixels_per_frame
            frame_pixels = [
                float(pixel_data_list[frame_start + k]) for k in range(pixels_per_frame)
            ]
            main_pix_array.extend(reversed(frame_pixels))

        main_pix_array = list(reversed(main_pix_array))

        # Accumulate
        if not summed_pixel_values:
            summed_pixel_values = main_pix_array
        else:
            for i in range(len(summed_pixel_values)):
                summed_pixel_values[i] += main_pix_array[i]

    return summed_pixel_values
