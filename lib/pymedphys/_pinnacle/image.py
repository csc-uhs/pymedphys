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

from pymedphys._dicom.orientation import IMAGE_ORIENTATION_MAP
from pymedphys._imports import numpy as np
from pymedphys._imports import pydicom

from .constants import GImplementationClassUID, GTransferSyntaxUID
from .pinnacle_metadata import apply_equipment_stamps, generate_pinn2dicom_uid

# Slice location sign: for head-first orientations DICOM z = -TablePosition,
# for feet-first DICOM z = +TablePosition. TablePosition is in cm; DICOM in mm.
_SLICE_Z_SIGN = {
    "HFS": -1,
    "HFP": -1,
    "FFS": +1,
    "FFP": +1,
}

# ImagePositionPatient x/y signs.  The first pixel is at the corner
# opposite to the row/column direction vectors.  The image centre sits
# at DICOM (0, 0) in the transverse plane, so:
#   IPP_x = row_dir_x * (-half_x_mm)   [negate because first pixel is
#            at the start, not end, of the row direction]
#   IPP_y = col_dir_y * (-half_y_mm)
# Because row_dir and col_dir only have one non-zero component each in
# the four standard positions, this reduces to a sign lookup.
_IPP_XY_SIGN = {
    #          (x_sign, y_sign)
    "HFS": (-1, -1),
    "HFP": (+1, +1),
    "FFS": (+1, -1),
    "FFP": (-1, +1),
}

# This function will create dicom image files for each slice using the
# condensed pixel data from file ImageSet_%s.img


def create_image_files(image, export_path):
    """Create DICOM image files from raw Pinnacle binary pixel data.

    Used when the ``ImageSet_N.DICOM/`` directory is absent and images
    must be reconstructed from ``ImageSet_N.img``.  Handles all standard
    patient orientations (HFS, HFP, FFS, FFP, and decubitus variants).
    """

    patient_info = image.pinnacle.patient_info
    image_header = image.image_header
    image_info = image.image_info
    image_set = image.image_set

    if not image_header:
        image.logger.error(
            "Cannot create image files: image header (.header) is missing"
        )
        return
    if not image_info:
        image.logger.error(
            "Cannot create image files: image info (.ImageInfo) is missing"
        )
        return

    currentpatientposition = image_header.get("patient_position", "HFS")

    # scan date/time from ImageSet file (may be missing in old archives)
    dateofscan = ""
    timeofscan = ""
    if image_set:
        dateofscan = image_set.get("scan_date", "")
        timeofscan = image_set.get("scan_time", "")

    modality = "CT"
    try:
        # Also should come from header file, but not always present
        modality = image_header["modality"]
    except KeyError:
        pass  # Incase it is not present in header

    img_file = os.path.join(image.path, f"ImageSet_{image.image['ImageSetID']}.img")
    if not os.path.isfile(img_file):
        image.logger.error(
            "Cannot create image files: raw image binary not found at %s",
            img_file,
        )
        return

    allframeslist = []
    pixel_array = np.fromfile(img_file, dtype=np.short)
    # will loop over every frame
    for i in range(0, int(image_header["z_dim"])):
        frame_array = pixel_array[
            i * int(image_header["x_dim"]) * int(image_header["y_dim"]) : (i + 1)
            * int(image_header["x_dim"])
            * int(image_header["y_dim"])
        ]
        allframeslist.append(frame_array)
    image.logger.debug("Length of frames list: %s", len(allframeslist))
    image.logger.debug(image_info[0])

    # The Pinnacle ImageInfo UIDs are internal identifiers that can collide
    # across patients when they share the same CT scan in Pinnacle.  On the
    # reconstructed-image path (no original DICOM) we generate fresh, unique
    # DICOM UIDs — using the same Pinn2Dicom algorithm / UID_ROOT that the
    # RT objects use when DICOM_EQUIPMENT.UID_ROOT is configured, or falling
    # back to pydicom's random generator otherwise.
    #
    # Updating the image_info dicts in place ensures that RT objects
    # (RTDOSE/RTPLAN/RTSTRUCT), which read UIDs via
    # plan.primary_image.image_info, automatically reference the new values.
    uid_root = getattr(image.pinnacle, "equipment_cfg", {}).get("UID_ROOT", "")

    if uid_root:
        new_study_uid = generate_pinn2dicom_uid(uid_root, "study")
        new_series_uid = generate_pinn2dicom_uid(uid_root, "series_ct")
        new_frame_uid = generate_pinn2dicom_uid(uid_root, "frame")
    else:
        new_study_uid = pydicom.uid.generate_uid()
        new_series_uid = pydicom.uid.generate_uid()
        new_frame_uid = pydicom.uid.generate_uid()

    for info in image_info:
        if uid_root:
            info["InstanceUID"] = generate_pinn2dicom_uid(uid_root, "ct")
        else:
            info["InstanceUID"] = pydicom.uid.generate_uid()
        info["StudyInstanceUID"] = new_study_uid
        info["SeriesUID"] = new_series_uid
        info["FrameUID"] = new_frame_uid

    image.logger.info(
        "Generated fresh DICOM UIDs for reconstructed images "
        "(root=%s) — StudyInstanceUID: %s, SeriesInstanceUID: %s",
        uid_root or "pydicom-random",
        new_study_uid,
        new_series_uid,
    )

    curframe = 0
    z_sign = _SLICE_Z_SIGN.get(currentpatientposition, -1)
    x_sign, y_sign = _IPP_XY_SIGN.get(currentpatientposition, (-1, -1))
    x_pixdim_mm = float(image_header["x_pixdim"]) * 10
    y_pixdim_mm = float(image_header["y_pixdim"]) * 10
    half_x_mm = x_pixdim_mm * float(image_header["x_dim"]) / 2
    half_y_mm = y_pixdim_mm * float(image_header["y_dim"]) / 2
    # ImagePositionPatient is the CENTRE of the first voxel (PS3.3 C.7.6.2.1),
    # i.e. half a pixel inside the grid corner. The previous half-extent value
    # (N/2 * pixdim) introduced a half-voxel origin offset.
    first_voxel_x_mm = half_x_mm - x_pixdim_mm / 2
    first_voxel_y_mm = half_y_mm - y_pixdim_mm / 2

    for info in image_info:
        sliceloc = z_sign * info["TablePosition"] * 10
        instuid = info["InstanceUID"]
        classuid = info["ClassUID"]
        slicenum = info["SliceNumber"]

        file_meta = pydicom.dataset.Dataset()
        file_meta.MediaStorageSOPClassUID = classuid
        file_meta.MediaStorageSOPInstanceUID = instuid
        file_meta.TransferSyntaxUID = GTransferSyntaxUID
        # this value remains static since implementation for creating
        # file is the same
        file_meta.ImplementationClassUID = GImplementationClassUID

        image_file_name = f"{modality}.{instuid}.dcm"
        ds = pydicom.dataset.FileDataset(
            image_file_name, {}, file_meta=file_meta, preamble=b"\x00" * 128
        )

        ds.SpecificCharacterSet = "ISO_IR 100"
        ds.ImageType = ["DERIVED", "PRIMARY", "AXIAL"]
        ds.AccessionNumber = ""
        ds.SOPClassUID = classuid
        ds.SOPInstanceUID = instuid
        ds.StudyDate = dateofscan
        ds.SeriesDate = dateofscan
        ds.AcquisitionDate = dateofscan
        ds.ContentDate = dateofscan
        ds.StudyTime = timeofscan
        ds.AcquisitionTime = timeofscan
        ds.Modality = modality
        ds.Manufacturer = ""  # Type 2; overwritten by apply_equipment_stamps
        ds.DerivationDescription = (
            "Reconstructed from Pinnacle binary image archive by Pinn2Dicom"
        )
        ds.StationName = modality

        # Apply site-specific equipment identification stamps from config.
        # For synthesised images there is no Pinnacle plan_info, so only
        # the configured values (manufacturer, institution, etc.) are set.
        equipment_cfg = getattr(image.pinnacle, "equipment_cfg", {})
        apply_equipment_stamps(ds, equipment_cfg)

        ds.PatientName = patient_info["FullName"]
        ds.PatientID = patient_info["MedicalRecordNumber"]
        ds.PatientBirthDate = patient_info["DOB"]
        ds.PatientSex = patient_info.get("Gender", "")[:1]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        # Archive does not reliably carry rescale values on this path;
        # -1024 / 1.0 is the near-universal CT convention.
        # Known limitation: unusual scanners with a different
        # intercept would be mis-scaled on the reconstruction path.
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1.0
        # ds.kvp = ?? This should be peak kilovoltage output of x ray
        # generator used
        ds.PatientPosition = currentpatientposition
        # this is probably x_pixdim * xdim = y_pixdim * ydim
        ds.DataCollectionDiameter = (
            float(image_header["x_pixdim"]) * 10 * float(image_header["x_dim"])
        )
        # ds.SpatialResolution = 0.35  # ???????
        # # ds.DistanceSourceToDetector = #???
        # # ds.DistanceSourceToPatient = #????
        # ds.GantryDetectorTilt = 0.0  # ??
        # ds.TableHeight = -158.0  # ??
        # ds.RotationDirection = "CW"  # ???
        # ds.ExposureTime = 1000  # ??
        # ds.XRayTubeCurrent = 398  # ??
        # ds.GeneratorPower = 48  # ??
        # ds.FocalSpots = 1.2  # ??
        # ds.ConvolutionKernel = "STND"  # ????
        ds.SliceThickness = float(image_header["z_pixdim"]) * 10
        ds.NumberOfSlices = int(image_header["z_dim"])
        # ds.StudyInstanceUID = studyinstuid
        # ds.SeriesInstanceUID = seriesuid
        ds.FrameOfReferenceUID = info["FrameUID"]
        ds.StudyInstanceUID = info["StudyInstanceUID"]
        ds.SeriesInstanceUID = info["SeriesUID"]
        # problem, some of these are repeated in image file so not sure
        # what to do with that
        ds.InstanceNumber = slicenum
        # first-voxel-centre geometry (see above).
        ds.ImagePositionPatient = [
            x_sign * first_voxel_x_mm,
            y_sign * first_voxel_y_mm,
            sliceloc,
        ]
        if currentpatientposition in IMAGE_ORIENTATION_MAP:
            ds.ImageOrientationPatient = [
                float(v) for v in IMAGE_ORIENTATION_MAP[currentpatientposition]
            ]
        else:
            # Unknown orientation — default to HFS and log a warning
            image.logger.warning(
                "Unknown patient position '%s' — defaulting ImageOrientationPatient "
                "to HFS",
                currentpatientposition,
            )
            ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PositionReferenceIndicator = ""  # Type 2; left empty unless known
        ds.SliceLocation = sliceloc
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = int(image_header["y_dim"])
        ds.Columns = int(image_header["x_dim"])
        # PixelSpacing is [row spacing (adjacent rows, i.e. the
        # y/vertical step), column spacing (adjacent columns, i.e. the
        # x/horizontal step)] per PS3.3 C.7.6.3.1; previously written in
        # [x, y] order, transposing the spacing for anisotropic grids.
        ds.PixelSpacing = [
            y_pixdim_mm,
            x_pixdim_mm,
        ]

        if curframe >= len(allframeslist):
            image.logger.error(
                "Frame index %d exceeds the %d frames decoded from the image "
                "binary; stopping image creation early. The slice count and "
                "binary frame count are inconsistent.",
                curframe,
                len(allframeslist),
            )
            break
        ds.PixelData = allframeslist[curframe].tobytes()

        output_file = os.path.join(export_path, image_file_name)
        image.logger.info("Creating image: %s", output_file)
        ds.save_as(output_file, enforce_file_format=True)
        curframe = curframe + 1


def convert_image(image, export_path):
    image.logger.debug(
        "Converting image patient name, birthdate and id to match pinnacle"
    )

    dicom_directory = os.path.join(
        image.path, f"ImageSet_{image.image['ImageSetID']}.DICOM"
    )

    if not os.path.exists(dicom_directory):
        image.logger.info("Dicom Image files do not exist. Creating image files")
        create_image_files(image, export_path)
        return

    # Verify the DICOM directory contains actual DICOM files.  Some older
    # archives (v7/v8) have an ImageSet_N.DICOM/ directory populated with
    # raw .img files instead of DICOM — detect this and fall back.
    dicom_files = []
    for file in os.listdir(dicom_directory):
        filepath = os.path.join(dicom_directory, file)
        if not os.path.isfile(filepath):
            continue
        try:
            test_ds = pydicom.dcmread(filepath, force=True, stop_before_pixels=True)
            if hasattr(test_ds, "SOPInstanceUID"):
                dicom_files.append(file)
        except Exception:
            pass

    if not dicom_files:
        image.logger.info(
            "DICOM directory exists but contains no valid DICOM files "
            "(%d files found). Falling back to creating image files.",
            len(os.listdir(dicom_directory)),
        )
        create_image_files(image, export_path)
        return

    patient_info = image.pinnacle.patient_info
    image_set = image.image_set

    for file in dicom_files:
        imageds = pydicom.dcmread(os.path.join(dicom_directory, file), force=True)

        imageds.PatientName = patient_info["FullName"]
        imageds.PatientID = patient_info["MedicalRecordNumber"]
        imageds.PatientBirthDate = patient_info["DOB"]
        imageds.PatientSex = patient_info.get("Gender", "")[:1]

        # Ensure required attributes are present — existing DICOM from
        # the Pinnacle archive may be missing these
        if "StudyTime" not in imageds and image_set:
            imageds.StudyTime = image_set.get("scan_time", "")
        if "SpecificCharacterSet" not in imageds:
            imageds.SpecificCharacterSet = "ISO_IR 100"

        preamble = getattr(imageds, "preamble", None)
        if not preamble:
            preamble = b"\x00" * 128

        output_file = os.path.join(
            export_path, f"{image.image['Modality']}.{imageds.SOPInstanceUID}.dcm"
        )

        imageds.save_as(output_file, enforce_file_format=True)
        image.logger.info("Exported: %s to %s", file, output_file)
