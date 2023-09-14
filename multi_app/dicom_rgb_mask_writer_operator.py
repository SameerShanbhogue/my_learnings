# This code is adapted from the MONAI deploy dicom_seg_writer_operator.py code 
# distributed under the Apache 2.0 license as described below :

# Copyright 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The code has been modified to output an RGB DICOM file to visualize segmentations 
# rather than DICOM SEG format in the original operator.



import datetime
import logging
import os
from pathlib import Path
from random import randint
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import numpy as np
from typeguard import typechecked

from monai.deploy.utils.importutil import optional_import
from monai.deploy.utils.version import get_sdk_semver
from monai.deploy.operators.dicom_utils import EquipmentInfo, ModelInfo, save_dcm_file, write_common_modules
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pydicom
from pydicom.uid import generate_uid
from PIL import Image as PIL_Image
from PIL import ImageOps

dcmread, _ = optional_import("pydicom", name="dcmread")
generate_uid, _ = optional_import("pydicom.uid", name="generate_uid")
ImplicitVRLittleEndian, _ = optional_import("pydicom.uid", name="ImplicitVRLittleEndian")
Dataset, _ = optional_import("pydicom.dataset", name="Dataset")
FileDataset, _ = optional_import("pydicom.dataset", name="FileDataset")
sitk, _ = optional_import("SimpleITK")
codes, _ = optional_import("pydicom.sr.codedict", name="codes")

if TYPE_CHECKING:
    import highdicom as hd
    from pydicom.sr.coding import Code
else:
    Code, _ = optional_import("pydicom.sr.coding", name="Code")
    hd, _ = optional_import("highdicom")

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, Image, InputContext, IOType, Operator, OutputContext
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries
from monai.deploy.core import AppContext



import datetime
import logging
import os
from pathlib import Path
from random import randint
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import numpy as np
from typeguard import typechecked

from monai.deploy.utils.importutil import optional_import
from monai.deploy.utils.version import get_sdk_semver

dcmread, _ = optional_import("pydicom", name="dcmread")
generate_uid, _ = optional_import("pydicom.uid", name="generate_uid")
ImplicitVRLittleEndian, _ = optional_import("pydicom.uid", name="ImplicitVRLittleEndian")
Dataset, _ = optional_import("pydicom.dataset", name="Dataset")
FileDataset, _ = optional_import("pydicom.dataset", name="FileDataset")
sitk, _ = optional_import("SimpleITK")
codes, _ = optional_import("pydicom.sr.codedict", name="codes")
if TYPE_CHECKING:
    import highdicom as hd
    from pydicom.sr.coding import Code
else:
    Code, _ = optional_import("pydicom.sr.coding", name="Code")
    hd, _ = optional_import("highdicom")

from monai.deploy.core import ConditionType, Fragment, Image, Operator, OperatorSpec
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries


class SegmentDescription:
    @typechecked
    def __init__(
        self,
        segment_label: str,
        segmented_property_category: Code,
        segmented_property_type: Code,
        algorithm_name: str,
        algorithm_version: str,
        algorithm_family: Code = codes.DCM.ArtificialIntelligence,
        tracking_id: Optional[str] = None,
        tracking_uid: Optional[str] = None,
        anatomic_regions: Optional[Sequence[Code]] = None,
        primary_anatomic_structures: Optional[Sequence[Code]] = None,
    ):
        """Class encapsulating the description of a segment within the segmentation.

        Args:
        segment_label: str
            User-defined label identifying this segment,
            DICOM VR Long String (LO) (see C.8.20-4
            https://dicom.nema.org/medical/Dicom/current/output/chtml/part03/sect_C.8.20.4.html
            "Segment Description Macro Attributes")
        segmented_property_category: pydicom.sr.coding.Code
            Category of the property the segment represents,
            e.g. ``Code("49755003", "SCT", "Morphologically Abnormal
            Structure")`` (see CID 7150
            http://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_7150.html
            "Segmentation Property Categories")
        segmented_property_type: pydicom.sr.coding.Code
            Property the segment represents,
            e.g. ``Code("108369006", "SCT", "Neoplasm")`` (see CID 7151
            http://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_7151.html
            "Segmentation Property Types")
        algorithm_name: str
            Name of algorithm used to generate the segment, also as the name assigned by a
            manufacturer to a specific software algorithm,
            DICOM VR Long String (LO) (see C.8.20-2
            https://dicom.nema.org/medical/dicom/2019a/output/chtml/part03/sect_C.8.20.2.html
            "Segmentation Image Module Attribute", and see 10-19
            https://dicom.nema.org/medical/dicom/2020b/output/chtml/part03/sect_10.16.html
            "Algorithm Identification Macro Attributes")
        algorithm_version: str
            The software version identifier assigned by a manufacturer to a specific software algorithm,
            DICOM VR Long String (LO) (see 10-19
            https://dicom.nema.org/medical/dicom/2020b/output/chtml/part03/sect_10.16.html
            "Algorithm Identification Macro Attributes")
        tracking_id: Optional[str], optional
            Tracking identifier (unique only with the domain of use).
        tracking_uid: Optional[str], optional
            Unique tracking identifier (universally unique) in the DICOM format
            for UIDs. This is only permissible if a ``tracking_id`` is also
            supplied. You may use ``pydicom.uid.generate_uid`` to generate a
            suitable UID. If ``tracking_id`` is supplied but ``tracking_uid`` is
            not supplied, a suitable UID will be generated for you.
        anatomic_regions: Optional[Sequence[pydicom.sr.coding.Code]], optional
            Anatomic region(s) into which segment falls,
            e.g. ``Code("41216001", "SCT", "Prostate")`` (see CID 4
            http://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_4.html
            "Anatomic Region", CID 403
            http://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_4031.html
            "Common Anatomic Regions", as as well as other CIDs for
            domain-specific anatomic regions)
        primary_anatomic_structures: Optional[Sequence[pydicom.sr.coding.Code]], optional
            Anatomic structure(s) the segment represents
            (see CIDs for domain-specific primary anatomic structures)
        """
        self._segment_label = segment_label
        self._segmented_property_category = segmented_property_category
        self._segmented_property_type = segmented_property_type
        self._tracking_id = tracking_id

        self._anatomic_regions = anatomic_regions
        self._primary_anatomic_structures = primary_anatomic_structures

        # Generate a UID if one was not provided
        if tracking_id is not None and tracking_uid is None:
            tracking_uid = hd.UID()
        self._tracking_uid = tracking_uid

        self._algorithm_identification = hd.AlgorithmIdentificationSequence(
            name=algorithm_name,
            family=algorithm_family,
            version=algorithm_version,
        )

    def to_segment_description(self, segment_number: int) -> hd.seg.SegmentDescription:
        """Get a corresponding highdicom Segment Description object.

        Args:
        segment_number: int
            Number of the segment. Must start at 1 and increase by 1 within a
            given segmentation object.

        Returns
        highdicom.seg.SegmentDescription:
            highdicom Segment Description containing the information in this
            object.
        """
        return hd.seg.SegmentDescription(
            segment_number=segment_number,
            segment_label=self._segment_label,
            segmented_property_category=self._segmented_property_category,
            segmented_property_type=self._segmented_property_type,
            algorithm_identification=self._algorithm_identification,
            algorithm_type="AUTOMATIC",
            tracking_uid=self._tracking_uid,
            tracking_id=self._tracking_id,
            anatomic_regions=self._anatomic_regions,
            primary_anatomic_structures=self._primary_anatomic_structures,
        )



#@md.input("seg_image", Image, IOType.IN_MEMORY)
#@md.input("study_selected_series_list", List[StudySelectedSeries], IOType.IN_MEMORY)
#@md.output("dicom_seg_instance", DataPath, IOType.DISK)
#@md.env(pip_packages=["pydicom >= 2.3.0", "highdicom >= 0.18.2"])
class DICOMRGBMaskWriterOperator(Operator):
    """
    This operator writes out a DICOM RGB image with segmentations overlaid
    """
    DEFAULT_OUTPUT_FOLDER = Path.cwd() / "output"
    # Supported input image format, based on extension. Intended for file based input.
    SUPPORTED_EXTENSIONS = [".nii", ".nii.gz", ".mhd"]
    # DICOM instance file extension. Case insensitive in string comparison.
    DCM_EXTENSION = ".dcm"

    def __init__(
        self,
        fragment: Fragment,
        *args,
        segment_descriptions: List[SegmentDescription],
        output_folder: Path,
        custom_tags: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Instantiates the DICOM Seg Writer instance with optional list of segment label strings.

        Each unique, non-zero integer value in the segmentation image represents a segment that must be
        described by an item of the segment descriptions list with the corresponding segment number.
        Items in the list must be arranged starting at segment number 1 and increasing by 1.

        For example, in the CT Spleen Segmentation application, the whole image background has a value
        of 0, and the Spleen segment of value 1. This then only requires the caller to pass in a list
        containing a segment description, which is used as label for the Spleen in the DICOM Seg instance.

        Note: this interface is subject to change. It is planned that a new object will encapsulate the
        segment label information, including label value, name, description etc.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            segment_descriptions: List[SegmentDescription]
                Object encapsulating the description of each segment present in the segmentation.
            output_folder: Folder for file output, overridden by named input on compute.
                           Defaults to current working dir's child folder, output.
            custom_tags: OptonalDict[str, str], optional
                Dictionary for setting custom DICOM tags using Keywords and str values only
        """

        self._seg_descs = [sd.to_segment_description(n) for n, sd in enumerate(segment_descriptions, 1)]
        self._custom_tags = custom_tags
        self.output_folder = output_folder if output_folder else DICOMSegmentationWriterOperator.DEFAULT_OUTPUT_FOLDER

        self.input_name_seg = "seg_image"
        self.input_name_series = "study_selected_series_list"
        self.input_name_output_folder = "output_folder"

        #self._omit_empty_frames = omit_empty_frames
        self.copy_tags = True
        self.model_info =  ModelInfo()
        self.equipment_info = EquipmentInfo()
        self.modality_type = "OT" #Other
        self.sop_class_uid = "1.2.840.10008.5.1.4.1.1.7" #secondary capture

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Set up the named input(s), and output(s) if applicable, aka ports.

        Args:
            spec (OperatorSpec): The Operator specification for inputs and outputs etc.
        """
        spec.input(self.input_name_seg)
        spec.input(self.input_name_series)
        spec.input(self.input_name_output_folder).condition(ConditionType.NONE)  # Optional input not requiring sender.

   

       
        
    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        """Performs computation for this operator and handles I/O.

        For now, only a single segmentation image object or file is supported and the selected DICOM
        series for inference is required, because the DICOM Seg IOD needs to refer to original instance.
        When there are multiple selected series in the input, the first series' containing study will
        be used for retrieving DICOM Study module attributes, e.g. StudyInstanceUID.

        Raises:
            FileNotFoundError: When image object not in the input, and segmentation image file not found either.
            ValueError: Neither image object nor image file's folder is in the input, or no selected series.
        """
        # Gets the input, prepares the output folder, and then delegates the processing.
        study_selected_series_list = op_input.receive(self.input_name_series)
        if not study_selected_series_list or len(study_selected_series_list) < 1:
            raise ValueError(f"Missing input, [{StudySelectedSeries}].")
        for study_selected_series in study_selected_series_list:
            if not isinstance(study_selected_series, StudySelectedSeries):
                raise ValueError(f"Element in input is not expected type, {StudySelectedSeries}.")

        seg_image = op_input.receive(self.input_name_seg)

        # In case the input is not the Image object, rather image file path.
        if not isinstance(seg_image, (Image, np.ndarray)) and (isinstance(seg_image, (Path, str))):
            seg_image_file, _ = self.select_input_file(str(seg_image))
            if Path(seg_image_file).is_file():
                seg_image = self._image_file_to_numpy(seg_image_file)
            else:
                raise ValueError("Input 'seg_image' is not an Image or a path.")

        # If the optional named input, output_folder, has content, use it instead of the one set on the object.
        # Since this input is optional, must check if data present and if Path or str.
        output_folder = None
        try:
            output_folder = op_input.receive(self.input_name_output_folder)
        except Exception:
            pass

        if not output_folder or not isinstance(output_folder, (Path, str)):
            output_folder = self.output_folder

        output_folder.mkdir(parents=True, exist_ok=True)
        self.process_images(seg_image, study_selected_series_list, output_folder)
    def process_images(
        self, image: Union[Image, Path], study_selected_series_list: List[StudySelectedSeries], output_dir: Path
    ):
        """ """

        if isinstance(image, Image):
            seg_image_numpy = image.asnumpy()
        elif isinstance(image, (Path, str)):
            seg_image_numpy = self._image_file_to_numpy(str(image))
        elif not isinstance(image, np.ndarray):
            raise ValueError("'image' is not a numpy array, Image object, or supported image file.")

        # Pick DICOM Series that was used as input for getting the seg image.
        # For now, first one in the list.
        for study_selected_series in study_selected_series_list:
            if not isinstance(study_selected_series, StudySelectedSeries):
                raise ValueError(f"Element in input is not expected type, {StudySelectedSeries}.")
            selected_series = study_selected_series.selected_series[0]
            dicom_series = selected_series.series
            self.create_dicom_rgb(seg_image_numpy, dicom_series, output_dir)
            break    
    

    def create_dicom_rgb(self, image: np.ndarray, dicom_series: DICOMSeries, output_dir: Path):
        
        #generate segmentation mask colormap
        new_prism= mpl.colormaps['prism']
        newcolors = new_prism(np.linspace(0, 1, 80)) # set to 80 colors
        black = np.array([0, 0, 0, 1])
        newcolors[0, :] = black
        newcmp = ListedColormap(newcolors)
        cm.register_cmap('newcmp',newcmp)
        cm.get_cmap('newcmp')
        
        
        
        if not output_dir.is_dir():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                raise ValueError("output_dir {output_dir} does not exist and failed to be created.") from None
        

        
        slices = dicom_series.get_sop_instances()
        
        ds = write_common_modules(
            dicom_series, self.copy_tags, self.modality_type, self.sop_class_uid , self.model_info, self.equipment_info
        )

        
        vol_data = np.stack([s.get_pixel_array() for s in slices], axis=0)
        vol_data = vol_data.astype(np.float32)
        
        # DICOM series setting
        series_UID= generate_uid()
        series_number=random_with_n_digits(4)
        series_desc =  "RGB_MASK(" + ds.SeriesDescription[:53] + ")"

        zdim = image.shape[0]
        for i in range(zdim):
            seg_sop_instance_uid = generate_uid()
            output_path = output_dir /  f"{seg_sop_instance_uid}{DICOMRGBMaskWriterOperator.DCM_EXTENSION}"

            raw_img = vol_data[i,:,:]
            seg_img = image[i,:,:]

            dcm = ds.copy()

            # Normalize the background (input) image
            background = 255 * ( 1.0 / raw_img.max() * (raw_img - raw_img.min()) )
            background = background.astype(np.ubyte)
            background_image = PIL_Image.fromarray(background).convert("RGB")

            seg_array = seg_img
            mask_image = PIL_Image.fromarray(np.uint8(newcmp(seg_array)*255)).convert("RGB")
        

            # Blend the two images
            final_image = PIL_Image.blend(mask_image, background_image, 0.75)
            final_array = np.array(final_image).astype(np.uint8) 
        
            
            #calc window settings for DICOM display
            window_min = np.amin(final_array)
            window_max =np.amax(final_array)
            window_middle = (window_max + window_min) / 2
            window_width = window_max - window_min
                
            # Write the final image back to a new DICOM (color) image 
            dcm.WindowCenter=f"{window_middle:.2f}" 
            dcm.WindowWidth=f"{window_width:.2f}"
            dcm.PixelSpacing= slices[i].get_native_sop_instance().PixelSpacing
            dcm.InstanceNumber = slices[i].get_native_sop_instance().InstanceNumber      
            dcm.SeriesInstanceUID = series_UID
            dcm.SeriesNumber = series_number
            dcm.SOPInstanceUID =seg_sop_instance_uid
            dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            dcm.Rows = final_image.height
            dcm.Columns = final_image.width
            dcm.PhotometricInterpretation = "RGB"
            dcm.SamplesPerPixel = 3
            dcm.BitsStored = 8
            dcm.BitsAllocated = 8
            dcm.HighBit = 7
            dcm.add_new(0x00280006, 'US', 0)
            dcm.is_little_endian = True
            dcm.fix_meta_info() 
            dcm.PixelData = final_array.tobytes()
            dcm.SeriesDescription = series_desc
            dcm.ImageType = "DERIVED\\SECONDARY"
            
            # Instance file name is the same as the new SOP instance UID with '_RGB' suffix
            save_dcm_file(dcm,output_path)
            try:
                # Test reading back
                _ = self._read_from_dcm(str(output_path))
            except Exception as ex:
                print("DICOM RBG mask creation failed. Error:\n{}".format(ex))
                raise

    def _read_from_dcm(self, file_path: str):
        """Read dcm file into pydicom Dataset

        Args:
            file_path (str): The path to dcm file
        """
        return dcmread(file_path)

    def select_input_file(self, input_folder, extensions=SUPPORTED_EXTENSIONS):
        """Select the input files based on supported extensions.

        Args:
            input_folder (string): the path of the folder containing the input file(s)
            extensions (array): the supported file formats identified by the extensions.

        Returns:
            file_path (string) : The path of the selected file
            ext (string): The extension of the selected file
        """

        def which_supported_ext(file_path, extensions):
            for ext in extensions:
                if file_path.casefold().endswith(ext.casefold()):
                    return ext
            return None

        if os.path.isdir(input_folder):
            for file_name in os.listdir(input_folder):
                file_path = os.path.join(input_folder, file_name)
                if os.path.isfile(file_path):
                    ext = which_supported_ext(file_path, extensions)
                    if ext:
                        return file_path, ext
            raise IOError("No supported input file found ({})".format(extensions))
        elif os.path.isfile(input_folder):
            ext = which_supported_ext(input_folder, extensions)
            if ext:
                return input_folder, ext
        else:
            raise FileNotFoundError("{} is not found.".format(input_folder))

    def _image_file_to_numpy(self, input_path: str):
        """Converts image file to numpy"""

        img = sitk.ReadImage(input_path)
        data_np = sitk.GetArrayFromImage(img)
        if data_np is None:
            raise RuntimeError("Failed to convert image file to numpy: {}".format(input_path))
        return data_np.astype(np.uint8)


def random_with_n_digits(n):
    assert isinstance(n, int), "Argument n must be a int."
    n = n if n >= 1 else 1
    range_start = 10 ** (n - 1)
    range_end = (10**n) - 1
    return randint(range_start, range_end)


def test():
    from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
    from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
    from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator

    current_file_dir = Path(__file__).parent.resolve()
    data_path = current_file_dir.joinpath("/input_data")
    out_dir = Path("/output_data").absolute()

    loader = DICOMDataLoaderOperator()
    series_selector = DICOMSeriesSelectorOperator()
    dcm_to_volume_op = DICOMSeriesToVolumeOperator()
    dicom_rgb_mask_writer = DICOMRGBMaskWriterOperator()

    # Testing with more granular functions
    study_list = loader.load_data_to_studies(data_path.absolute())
    series = study_list[0].get_all_series()[0]

    dcm_to_volume_op.prepare_series(series)
    voxels = dcm_to_volume_op.generate_voxel_data(series)
    metadata = dcm_to_volume_op.create_metadata(series)
    image = dcm_to_volume_op.create_volumetric_image(voxels, metadata)
    # Very crude thresholding
    image_numpy = (image.asnumpy() > 400).astype(np.uint8)

    dicom_rgb_mask_writer.create_dicom_rgb(image_numpy, series, out_dir)

    # Testing with the main entry functions
    study_list = loader.load_data_to_studies(data_path.absolute())
    study_selected_series_list = series_selector.filter(None, study_list)
    image = dcm_to_volume_op.convert_to_image(study_selected_series_list)
    # Very crude thresholding
    image_numpy = (image.asnumpy() > 400).astype(np.uint8)
    image = Image(image_numpy)
    dicom_rgb_mask_writer.process_images(image, study_selected_series_list, out_dir)


if __name__ == "__main__":
    test()