from __future__ import annotations
from typing import Dict, List, Tuple
import sofar
import glob
import numpy as np
from imageio.v3 import imread
import os
import torch
import tqdm
from metrics import MeanSpectralDistortion


all_tasks = [np.arange(19).tolist(), np.arange(19, step=3).tolist(), [3, 6, 9]]


class SonicomDatabase(torch.utils.data.Dataset):

    def __init__(
        self,
        root_dir: str,
        hrtf_type="FreeFieldCompMinPhase",
        no_itd=True,
        sampling_rate="48kHz",
        nfft=256,
        training_data: bool = True,
        task_id: int = 0,
        folder_structure: str = 'v2'
    ):
        """
        Args:
            root_dir: Directory with all the HRTF files in subfolder.
            hrtf_type: can be any of ['Raw','Windowed','FreeFieldComp','FreeFieldCompMinPhase'(default)]
            sampling_rate: any of 44kHz, 48kHz, 96kHz
            nfft: fft length
            training_data: if true then return training dataset
            task_id: task id determines how many images will be used for inference. Can be 0, 1, or 2. 
        """
        super().__init__()
        self.root_dir = root_dir
        self.hrtf_type = hrtf_type
        self.nfft = nfft

        if no_itd:
            itd_str = "NoITD_"
        else:
            itd_str = ""

        if folder_structure not in {'v1', 'v2'}:
            raise RuntimeError('Unknown folder structure version')
        pathname = f"P*/P*/HRTF/HRTF/{sampling_rate}/*_{hrtf_type}_{itd_str}{sampling_rate}.sofa" if folder_structure == 'v1' else \
                   f"SONICOM_HRTF/P*/HRTF/HRTF/{sampling_rate}/*_{hrtf_type}_{itd_str}{sampling_rate}.sofa"
        self.hrtf_files = glob.glob(os.path.join(root_dir, pathname))
        # print('Found ' + str(len(self.hrtf_files)) + ' files')


        if training_data:
            self.image_dir = os.path.join(root_dir, "SONICOM_TrainingData_pics")
            self.task = all_tasks[0]
        else:
            print(os.path.join(root_dir, "SONICOM_TestData_pics"))
            self.image_dir = os.path.join(root_dir, "SONICOM_TestData_pics")
            self.task = all_tasks[task_id]

        self.all_image_names = [i for i in os.listdir(self.image_dir) if ".png" in i]
        self.all_subjects = self.get_available_ids()

        # read one to get coordinate system information
        try:
            tmp = sofar.read_sofa(self.hrtf_files[0], verbose=False)
            self.training_data = training_data
            self.position = tmp.SourcePosition
        except (IndexError, ValueError):
            print("Check if Dataset is saved as described in the notebook.")
            return None

    def __len__(self):
        return len(self.all_subjects)

    def load_all_hrtfs(self) -> torch.Tensor:
        """
        This function loads all the HRTFs from the list of IDs.

        Returns:
            Magnitude Spectrum of HRTFs : torch.Tensor
        """
        HRTFs = torch.zeros(
            (self.__len__(), self.position.shape[0], 2, self.nfft // 2 + 1)
        )

        allids = np.unique([cur_id[:5] for cur_id in self.all_image_names])
        for idx in range(len(allids)):
            if allids[idx] == allids[idx - 1] and idx > 0:
                HRTFs[idx] = HRTFs[idx - 1]
            else:
                HRTFs[idx] = torch.from_numpy(
                    self.load_subject_id_hrtf(allids[idx])
                ).abs()
        return HRTFs

    def load_image(self, image_name: str) -> Tuple[np.ndarray, str, str]:
        """
        This function read all the image files in the directory, get the ID of the image, Left or Right side of the pinna.

        Args:
            image_name (str): e.g. P0002_left_0.png

        Returns:
            image: torch.Tensor
            ID: (str) Subject ID of the loaded image
            Face_Side: (str) If the image loaded is of the left ear or the right ear
        """

        image = imread(os.path.join(self.image_dir, image_name))
        ID = image_name[:5]
        Face_Side = ["left" if "left" in image_name else "right"]

        return image, ID, Face_Side

    def get_image_names_from_id(self, id: str) -> List[str]:
        """
        This function helps to get the image names from the directory.

        Args:
            id (str): Subject ID e.g. 'P0001'
        Returns:
            List of image name
        """
        return [
            x for x in os.listdir(self.image_dir) if f"{id}" in x
        ]  # glob.glob(os.path.join(self.image_dir, f'{id}*'))

    def get_available_ids(self) -> List[str]:
        """
        This function returns all unique IDs from the list of images.

        Args:
            all_images (list of str)
        Returns:
            list of unique IDs
        """
        return list({name[:5] for name in self.all_image_names})

    def _extract_number_of_image(self, image_name: str) -> List[int]:
        """
        Extracts the image number of the subject from an image filename.

        Args:
            image_name (str): Filename of the image.

        Returns:
            Optional[int]: value if successfully extracted; otherwise, None.
        """
        try:
            azi_str = image_name.split("t_")[1]
            number = int(azi_str.split(".")[0])
            return number
        except (IndexError, ValueError):
            return None

    def _get_task_subset_image_names(self, image_names: List[str]) -> Tuple[List[str], List[str]]:
        """
        Returns two Lists of left and right image names from selected subset (based on task).

        Args:
            image_names (List of str): Filenames of the images.

        Returns:
            Dict e.g. {left_0: [0, 'P0002_left_0.png'], right_0: [1, 'P0002_right_0.png']}
        """

        left_names = []
        right_names = []
        for i in image_names:
            cur_azi = self._extract_number_of_image(i)
            if cur_azi in self.task:
                if "left" in i:
                    left_names.append(i)  # channel, name
                if "right" in i:
                    right_names.append(i)

        return left_names, right_names

    def get_all_images_and_HRTF_from_id(self, id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads all the images for the subject (only the subset) and the corresponding HRTF.

        Args:
            ID of each subject (str): e.g. P0001

        Returns:
            all_images: torch.Tensor of shape (ear_idx, image_idx, height, width)
            HRTFs: torch.Tensor
        """
        image_names = self.get_image_names_from_id(id)
        image_names.sort()
        left_images_filenames, right_images_filenames = (
            self._get_task_subset_image_names(image_names)
        )
        left_images = []
        right_images = []

        if not left_images_filenames or not right_images_filenames:
            raise FileNotFoundError(f"No images found for subject ID '{id}'.")        
        
        left_images = torch.from_numpy(np.stack([imread(os.path.join(self.image_dir, path)) for path in left_images_filenames]))
        right_images = torch.from_numpy(np.stack([imread(os.path.join(self.image_dir, path)) for path in right_images_filenames]))

        all_images = torch.stack((left_images, right_images))

        HRTF = self.load_subject_id_hrtf(id)

        return all_images, HRTF

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used by the Dataloader, it iterates through the number of subjects in the current dataset
        and provides the corresponding Images, HRTFs and Subject IDs.
        """
        id = self.all_subjects[idx]
        all_images, HRTF = self.get_all_images_and_HRTF_from_id(id)

        return all_images, HRTF

    def load_subject_id_hrtf(self, subject_id: str, return_sofa: bool = False) -> sofar.Sofa | np.ndarray:
        """
        This function load the HRIR data for the given file name and compute the RFFT of the HRIRs then return HRTFs
        Example if the file name is P0001, this function will load the sofa file of P0001 - read it and return the HRTF data of the P001

        Args:
            subject_id (str): e.g. P0001, ..., P0200
        """

        hrtf_file = [s for s in self.hrtf_files if subject_id + "_" + self.hrtf_type in s][0]
        if not hrtf_file:
            print(subject_id + " Not found!")
            return None
        if return_sofa:
            return sofar.read_sofa(hrtf_file, verbose=False)
        else:
            hrir = self._load_hrir(hrtf_file)
            return self._compute_HRTF(hrir)

    def _load_hrir(self, hrtf_file: sofar.Sofa) -> np.ndarray:
        """
        This function load the HRIR data for the given filename.

        Args:
              sofa file
            Returns:
               HRIR data"""
        data = sofar.read_sofa(hrtf_file, verbose=False)
        return data.Data_IR

    def _compute_HRTF(self, hrir: np.ndarray) -> np.ndarray:
        """
        This function compute the RFFT of the given HRIRs and return HRTFs.

        Args:
              HRIRs (time domain)
            Returns:
               HRTFs (Frequency domain)"""

        return np.fft.rfft(hrir, n=self.nfft)


def baseline_spectral_distortion(sd: SonicomDatabase, path_to_baseline_hrtf: str = "./data/Average_HRTFs.sofa") -> float:
    # this function calculate the spectral difference as mean square error between your ground truth HRTFs and the baseline average HRTFs
    # load all HRTFS, concat in 1 tensor, clone Average_HRTFs as many times and then find get_spectral_distortion
    """Returns:
    baseline prediction MSE in dB
    """

    all_HRTFs = sd.load_all_hrtfs()
    baseline_HRIR = sofar.read_sofa(path_to_baseline_hrtf, verbose=False).Data_IR
    baseline_HRTF = torch.from_numpy(sd._compute_HRTF(baseline_HRIR))
    baseline_HRTF = baseline_HRTF.unsqueeze(0).repeat(all_HRTFs.shape[0], 1, 1, 1)
    eval_metric = MeanSpectralDistortion()

    return eval_metric.get_spectral_distortion(all_HRTFs, baseline_HRTF)


def convert_to_HRIR(hrtfs: np.ndarray) -> np.ndarray:
    return np.fft.irfft(hrtfs, axis=-1)

def save_sofa(HRIR: np.ndarray, output_path: str, reference_sofa: sofar.Sofa):
    """
    Save the HRIR to a SOFA object file. See main() for example usage

    Args:
        HRIR (np.ndarray): HRIR of shape (793, 2, 256).
        output_path (str): Path where the SOFA file will be saved.
        reference_sofa (str): The SOFA object to copy information
    """
    hrtf = reference_sofa
    hrtf.Data_IR = HRIR
    sofar.write_sofa(output_path, hrtf, 0)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    sonicom_root = "./data"
    sd = SonicomDatabase(sonicom_root, training_data=False, task_id=2)
    train_dataloader = DataLoader(sd, batch_size=1, shuffle=False)

    for i, (images, hrtf) in tqdm.tqdm(enumerate(train_dataloader)):
        print(f"Image size: {images.shape} and HRTF size: {hrtf.shape}")
        break

    Error = baseline_spectral_distortion(sd)
    print(Error)
