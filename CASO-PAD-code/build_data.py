import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import functional as F
import random
from typing import Tuple, List, Optional

from utils import get_dataset_path



######################## Domain IDs ###########################

DOMAIN_IDS = {
    "OULU": 0,
    "RA": 1,
    "CASIA": 2,
    "MSU": 3,
    "SIW": 4,
    "RY": 5,
    "RM": 6,
}


##########################################################
####### Dataset class for OULU-NPY dataset #######

class VideoDataset_Oulu(Dataset):
    def __init__(self, orig_root_dir, file_list_path, transform=None, num_frames=16, is_train=False, split=None, protocol=None):
        """
        Args:
            orig_root_dir (str): Path to the root directory containing the original video files.
            depth_root_dir (str): Path to the root directory containing the corresponding depth video files.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_frames (int, optional): Number of frames to be sampled from each video.
            is_train (bool, optional): Flag to indicate if the dataset is used for training.
        """
        self.orig_root_dir = orig_root_dir
        self.file_list_path = file_list_path
        self.transform = transform
        self.num_frames = num_frames
        self.is_train = is_train
        self.classes = ['attack', 'real']  # Label mapping: 0 = attack, 1 = real
        self.split = split
        self.protocol = protocol
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Load paths to original videos and their corresponding depth videos, along with labels.
        """
        samples = []

        if self.protocol!='all':
            # Read the file list and extract filenames and labels
            with open(self.file_list_path, 'r') as f:
                lines = f.readlines()

            valid_files = {}  # Dictionary to store file names and labels
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) != 2:
                    continue  # Skip malformed lines
                
                label = 1 if parts[0].strip() == "+1" else 0
                filename = parts[1].strip()
                valid_files[filename] = label  # Store without extension

            # Loop through files for the user (Phone_Session_User_File.avi)
            for file_name in os.listdir(self.orig_root_dir):
                if file_name.endswith(('.avi', '.mp4', '.mov')):
                    base_name, ext = os.path.splitext(file_name)  # Get filename without extension
                    if base_name in valid_files:

                        video_path = os.path.join(self.orig_root_dir, file_name)

                        # Extract access type from the file name
                        access_type = int(file_name.split('_')[-1].split('.')[0])

                        # Determine label based on access type (1 = real, 2-5 = attack)
                        label = 1 if access_type == 1 else 0

                        samples.append((video_path, label, access_type))

        else:
            print('All data being used... No OULU-NPU Protocol Applied\n')

            for file_name in os.listdir(self.orig_root_dir):
                if file_name.endswith(('.avi', '.mp4', '.mov')):
                    video_path = os.path.join(self.orig_root_dir, file_name)

                    # Extract access type from the file name
                    access_type = int(file_name.split('_')[-1].split('.')[0])

                    # Determine label based on access type (1 = real, 2-5 = attack)
                    label = 1 if access_type == 1 else 0

                    samples.append((video_path, label, access_type))
                        
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, video_path, frame_indices, is_depth=False):
        """
        Load frames from the specified video file at the given frame indices.
        For depth videos, convert frames to single-channel grayscale.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        for idx in frame_indices:
            if idx >= total_frames:  # Ensure indices do not exceed total frames
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            if is_depth:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                frame = np.expand_dims(frame, axis=-1)  # Add channel dimension

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if not is_depth else frame
            frames.append(F.to_tensor(frame))

        cap.release()

        # # Handle the case where no frames were loaded
        # if not frames:
        #     print(f"No frames loaded for video: {video_path}, indices: {frame_indices}")
        #     with open(os.path.join(args.log_dir, 'training_log.txt'), 'a') as log_file:
        #         log_file.write(f"No frames loaded for video: {video_path}, indices: {frame_indices}")

        #     # Handle empty frames list by appending a black frame of the expected size
        #     placeholder_frame = np.zeros((224, 224, 1 if is_depth else 3), dtype=np.uint8)
        #     frames.append(F.to_tensor(placeholder_frame))

        # Pad if fewer frames are available
        while len(frames) < len(frame_indices):
            frames.append(frames[-1])

        return frames

    def __getitem__(self, idx):
        """
        Override __getitem__ to load frames from both original and depth videos
        with synchronized indices.
        """
        # orig_video_path, depth_video_path, label = self.samples[idx]
        orig_video_path, label, access_type = self.samples[idx]

        cap = cv2.VideoCapture(orig_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Determine frame indices
        if self.is_train:
            start_frame = np.random.randint(0, max(1, total_frames - self.num_frames + 1))
        else:
            start_frame = 0

        frame_indices = np.linspace(start_frame, start_frame + self.num_frames - 1, self.num_frames, dtype=int)

        # Load frames from both videos using the same frame indices
        orig_frames = self._load_frames(orig_video_path, frame_indices, is_depth=False)
        # depth_frames = self._load_frames(depth_video_path, frame_indices, is_depth=True)        
        
        # Apply transformations to both original and depth frames
        if self.transform:
            orig_frames = [self.transform(frame) for frame in orig_frames]
            # depth_frames = [self.transform(frame) for frame in depth_frames]

        # Apply augmentation if in training mode
        if self.is_train:
            angle, scale = self._random_augmentation_params()
            orig_frames = [self.apply_augmentation(frame, angle, scale) for frame in orig_frames]
            # depth_frames = [self.apply_augmentation(frame, angle, scale) for frame in depth_frames]

        orig_frames = torch.stack(orig_frames)  # Shape: (num_frames, 3, H, W)
        # depth_frames = torch.stack(depth_frames)  # Shape: (num_frames, 1, H, W)

        # # Combine into a 4-channel tensor
        # combined_frames = torch.cat([orig_frames, depth_frames], dim=1)  # Shape: (num_frames, 4, H, W)
        combined_frames = orig_frames

        if self.split == "test":
            return {'img':combined_frames, 'label':label, 'access_type':access_type, "dataset":DOMAIN_IDS["OULU"]}
        else:
            return {'img':combined_frames, 'label':label, "dataset":DOMAIN_IDS["OULU"]}

        
    def _random_augmentation_params(self):
        """
        Generate random augmentation parameters (angle and scale) for training.
        """
        angle = random.uniform(-180, 180) if random.random() > 0.5 else 0
        scale = random.uniform(0.7, 1.3) if random.random() > 0.5 else 1
        return angle, scale

    def apply_augmentation(self, image, angle, scale):
        """Apply rotation and scaling augmentation."""
        if angle != 0:
            image = F.rotate(image, angle)
        if scale != 1:
            image = F.affine(image, angle=0, translate=(0, 0), scale=scale, shear=0)
        return image


##########################################################
####### Dataset class for RA, RY, RM #######

class VideoDataset(Dataset):
    def __init__(self, orig_root_dir, transform=None, num_frames=16, is_train=False, datasetname=None):
        """
        Args:
            orig_root_dir (str): Path to the root directory containing the original video files.
            depth_root_dir (str): Path to the root directory containing the corresponding depth video files.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_frames (int, optional): Number of frames to be sampled from each video.
            is_train (bool, optional): Flag to indicate if the dataset is used for training.
        """
        self.orig_root_dir = orig_root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.is_train = is_train
        self.datasetname = datasetname
        self.classes = ['attack', 'real']  # Label mapping: 0 = attack, 1 = real
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Load paths to original videos and their corresponding depth videos, along with labels.
        """
        samples = []
        for cls in self.classes:
            orig_cls_dir = os.path.join(self.orig_root_dir, cls)
            for root, _, files in os.walk(orig_cls_dir):
                for fname in files:
                    if fname.endswith(('.mp4', '.mov', '.avi')):
                        orig_video_path = os.path.join(root, fname)
                        samples.append((orig_video_path, self.classes.index(cls)))
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, video_path, frame_indices, is_depth=False):
        """
        Load frames from the specified video file at the given frame indices.
        For depth videos, convert frames to single-channel grayscale.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        for idx in frame_indices:
            if idx >= total_frames:  # Ensure indices do not exceed total frames
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            if is_depth:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                frame = np.expand_dims(frame, axis=-1)  # Add channel dimension

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if not is_depth else frame
            frames.append(F.to_tensor(frame))

        cap.release()

        # # Handle the case where no frames were loaded
        # if not frames:
        #     print(f"No frames loaded for video: {video_path}, indices: {frame_indices}")
        #     with open(os.path.join(args.log_dir, 'training_log.txt'), 'a') as log_file:
        #         log_file.write(f"No frames loaded for video: {video_path}, indices: {frame_indices}")

        #     # Handle empty frames list by appending a black frame of the expected size
        #     placeholder_frame = np.zeros((224, 224, 1 if is_depth else 3), dtype=np.uint8)
        #     frames.append(F.to_tensor(placeholder_frame))

        # Pad if fewer frames are available
        while len(frames) < len(frame_indices):
            frames.append(frames[-1])

        return frames

    def __getitem__(self, idx):
        """
        Override __getitem__ to load frames from both original and depth videos
        with synchronized indices.
        """
        # orig_video_path, depth_video_path, label = self.samples[idx]
        orig_video_path, label = self.samples[idx]

        cap = cv2.VideoCapture(orig_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Determine frame indices
        if self.is_train:
            start_frame = np.random.randint(0, max(1, total_frames - self.num_frames + 1))
        else:
            start_frame = 0

        frame_indices = np.linspace(start_frame, start_frame + self.num_frames - 1, self.num_frames, dtype=int)

        # Load frames from both videos using the same frame indices
        orig_frames = self._load_frames(orig_video_path, frame_indices, is_depth=False)
        # depth_frames = self._load_frames(depth_video_path, frame_indices, is_depth=True)        
        
        # Apply transformations to both original and depth frames
        if self.transform:
            orig_frames = [self.transform(frame) for frame in orig_frames]
            # depth_frames = [self.transform(frame) for frame in depth_frames]

        # Apply augmentation if in training mode
        if self.is_train:
            angle, scale = self._random_augmentation_params()
            orig_frames = [self.apply_augmentation(frame, angle, scale) for frame in orig_frames]
            # depth_frames = [self.apply_augmentation(frame, angle, scale) for frame in depth_frames]

        orig_frames = torch.stack(orig_frames)  # Shape: (num_frames, 3, H, W)
        # depth_frames = torch.stack(depth_frames)  # Shape: (num_frames, 1, H, W)

        # # Combine into a 4-channel tensor
        # combined_frames = torch.cat([orig_frames, depth_frames], dim=1)  # Shape: (num_frames, 4, H, W)
        combined_frames = orig_frames

        return {'img':combined_frames, 'label':label, "dataset":DOMAIN_IDS[self.datasetname]}
        
        
    def _random_augmentation_params(self):
        """
        Generate random augmentation parameters (angle and scale) for training.
        """
        angle = random.uniform(-180, 180) if random.random() > 0.5 else 0
        scale = random.uniform(0.7, 1.3) if random.random() > 0.5 else 1
        return angle, scale

    def apply_augmentation(self, image, angle, scale):
        """Apply rotation and scaling augmentation."""
        if angle != 0:
            image = F.rotate(image, angle)
        if scale != 1:
            image = F.affine(image, angle=0, translate=(0, 0), scale=scale, shear=0)
        return image


##########################################################
####### Dataset class for SiW dataset #######

class SIW_Dataset(Dataset):
    """
    Standalone SIW dataset class (Protocol 1).

    Classes:
        Spoof -> 0
        Live  -> 1

    Directory structure:
        root/
            Spoof/**.mov
            Live/**.mov
            protocol_files/*.txt
    """

    def __init__(
        self,
        orig_root_dir: str,
        transform=None,
        num_frames: int = 16,
        is_train: bool = False,
        target_size: Tuple[int, int] = (224, 224),
        protocol: Optional[str] = '1',
        split: str = "train",   # train | val | test
        seed: int = 1234,
    ):
        super().__init__()

        assert protocol == '1', "Only SIW Protocol-1 is implemented"
        assert split in ("train", "val", "test")

        self.orig_root_dir = orig_root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.is_train = is_train
        self.target_size = target_size
        self.protocol = protocol
        self.split = split
        self.seed = seed

        self.classes = ("Spoof", "Live")
        self.samples = self._load_samples()

        print(f"SIW Dataset (split={self.split}, is_train={self.is_train}) initialized!")
        print(f"Total samples: {len(self.samples)}")

    # ===============================
    # Sample list construction
    # ===============================

    def _load_samples(self):
        samples = []
        proto_dir = os.path.join(self.orig_root_dir, "protocol_files")

        train_spoof = read_txt_list(os.path.join(proto_dir, "trainlist_all.txt"))
        train_live  = read_txt_list(os.path.join(proto_dir, "trainlist_live.txt"))
        test_spoof  = read_txt_list(os.path.join(proto_dir, "testlist_all.txt"))
        test_live   = read_txt_list(os.path.join(proto_dir, "testlist_live.txt"))

        if self.split == "test":
            spoof_ids = test_spoof
            live_ids = test_live
        elif self.split == "train":
            spoof_ids = split_train_val(train_spoof, is_train=True, seed=self.seed)
            live_ids = split_train_val(train_live, is_train=True, seed=self.seed)
        else:  # val
            spoof_ids = split_train_val(train_spoof, is_train=False, seed=self.seed)
            live_ids = split_train_val(train_live, is_train=False, seed=self.seed)

        class_map = {
            "Spoof": spoof_ids,
            "Live":  live_ids,
        }

        for cls_idx, cls_name in enumerate(self.classes):
            for stem in class_map[cls_name]:
                vpath = find_video_by_stem(self.orig_root_dir, cls_name, stem)
                if vpath is None:
                    continue

                if cls_name == "Spoof":
                    spoof_type = os.path.basename(os.path.dirname(vpath))
                else:
                    spoof_type = "Live"

                samples.append((vpath, cls_idx, spoof_type))

        return sorted(samples)

    # ===============================
    # Dataset API
    # ===============================

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, video_path, frame_indices):
        """
        Load RGB frames from the specified video file at the given frame indices.
        Matches the reference class behavior:
        - reads frames using provided indices
        - stops if reading fails
        - pads using the last valid frame until num_frames is reached
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        for idx in frame_indices:
            if idx >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(F.to_tensor(frame))   # (3, H, W)

        cap.release()

        # fallback if nothing could be loaded
        if not frames:
            h, w = self.target_size
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            frames.append(F.to_tensor(placeholder))

        # pad to num_frames using last valid frame
        while len(frames) < len(frame_indices):
            frames.append(frames[-1])

        return frames

    def __getitem__(self, idx):
        video_path, label, spoof_type = self.samples[idx]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # same frame sampling style as reference class
        if self.is_train:
            start_frame = np.random.randint(0, max(1, total_frames - self.num_frames + 1))
        else:
            start_frame = 0

        frame_indices = np.linspace(
            start_frame,
            start_frame + self.num_frames - 1,
            self.num_frames,
            dtype=int
        )

        # 1) load frames
        rgb_frames = self._load_frames(video_path, frame_indices)

        # 2) apply transform first (same as reference)
        if self.transform:
            rgb_frames = [self.transform(frame) for frame in rgb_frames]

        # 3) apply same augmentation params to all frames if training
        if self.is_train:
            angle, scale = self._random_augmentation_params()
            rgb_frames = [self.apply_augmentation(frame, angle, scale) for frame in rgb_frames]

        # 4) stack frames at end
        rgb_frames = torch.stack(rgb_frames)   # (num_frames, 3, H, W)

        if self.split == "test":
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            return {
                'img': rgb_frames,
                'label': label,
                'access_type': (video_id, spoof_type)
                }
        return {
            'img': rgb_frames,
            'label': label
            }

    # ===============================
    # Augmentations
    # ===============================

    def _random_augmentation_params(self):
        """
        Same logic as reference class.
        """
        angle = random.uniform(-180, 180) if random.random() > 0.5 else 0
        scale = random.uniform(0.7, 1.3) if random.random() > 0.5 else 1
        return angle, scale

    def apply_augmentation(self, image, angle, scale):
        """
        Same logic as reference class.
        """
        if angle != 0:
            image = F.rotate(image, angle)
        if scale != 1:
            image = F.affine(image, angle=0, translate=(0, 0), scale=scale, shear=0)
        return image


# ==========================================================
# MSU-MFSD dataset class
# ==========================================================

class MSU_MFSD_Dataset(Dataset):
    """
    MSU-MFSD face spoofing dataset loader.

    Labels:
        real   -> 1
        attack -> 0

    Supported protocols:
        grand / all
        1 / ipad_android
        2 / iphone_android
        3 / print_android
        4 / ipad_laptop
        5 / iphone_laptop
        6 / print_laptop

    Split handling:
        train -> subjects from train_sub_list.txt
        val   -> deterministic 80/20 split of train subjects
        test  -> subjects from test_sub_list.txt
    """

    PROTOCOL_MAP = {
        "1": ("android", "ipad_video"),
        "2": ("android", "iphone_video"),
        "3": ("android", "printed_photo"),
        "4": ("laptop", "ipad_video"),
        "5": ("laptop", "iphone_video"),
        "6": ("laptop", "printed_photo"),
        "ipad_android": ("android", "ipad_video"),
        "iphone_android": ("android", "iphone_video"),
        "print_android": ("android", "printed_photo"),
        "ipad_laptop": ("laptop", "ipad_video"),
        "iphone_laptop": ("laptop", "iphone_video"),
        "print_laptop": ("laptop", "printed_photo"),
    }

    def __init__(
        self,
        orig_root_dir: str,
        transform=None,
        num_frames: int = 16,
        is_train: bool = False,
        target_size: Tuple[int, int] = (224, 224),
        protocol: Optional[str] = "grand",
        split: str = "train",   # train | val | test
        seed: int = 1234,
    ):
        super().__init__()

        # assert split in ("train", "val", "test"), "split must be train/val/test"
        assert split in ("train", "val", "test", "train80", "valtrain20"), (
            "split must be one of: "
            "train, val, test, train80, valtrain20"
        )

        self.orig_root_dir = orig_root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.is_train = is_train
        self.target_size = target_size
        self.split = split
        self.seed = seed
        self.protocol = self._normalize_protocol(protocol)

        self.scene_root = self._resolve_scene_root(self.orig_root_dir)
        self.real_dir = os.path.join(self.scene_root, "real")
        self.attack_dir = os.path.join(self.scene_root, "attack")

        # print(f"\n >>>>>>>>>> [DEBUG] MSU root      : {self.orig_root_dir}")
        # print(f"\n >>>>>>>>>> [DEBUG] MSU scene root: {self.scene_root}")
        # print(f"\n >>>>>>>>>> [DEBUG] Real dir      : {self.real_dir}")
        # print(f"\n >>>>>>>>>> [DEBUG] Attack dir    : {self.attack_dir}")

        self.train_list_path = os.path.join(self.orig_root_dir, "train_sub_list.txt")
        self.test_list_path = os.path.join(self.orig_root_dir, "test_sub_list.txt")

        self.samples = self._load_samples()

        print(f"MSU-MFSD Dataset (split={self.split}, protocol={self.protocol}, is_train={self.is_train}) initialized!")
        print(f"Total samples: {len(self.samples)}")

    def _normalize_protocol(self, protocol):
        if protocol is None:
            return "grand"
        p = str(protocol).strip().lower()
        if p in ("all", "grand", "g"):
            return "grand"
        return p

    def _resolve_scene_root(self, root_dir: str) -> str:
        """
        Accept either:
            root/
                scene01/
                    real/
                    attack/
        or directly:
            root/scene01/
        """
        if os.path.isdir(os.path.join(root_dir, "scene01")):
            return os.path.join(root_dir, "scene01")
        if os.path.basename(os.path.normpath(root_dir)) == "scene01":
            return root_dir
        return root_dir

    def _parse_msu_stem(self, stem: str):

        parts = stem.split("_")

        if len(parts) < 5:
            return None

        kind = parts[0].lower()

        ##########################################################
        # real_client001_android_SD_scene01
        ##########################################################

        if kind == "real":

            # subject_id = parts[1].replace("client", "")
            subject_id = str(int(parts[1].replace("client", "")))

            return {
                "kind": "real",
                "subject_id": subject_id,
                "camera_type": parts[2].lower(),
                "resolution": parts[3].lower(),
                "attack_type": "real",
                "scene": parts[4].lower(),
            }

        ##########################################################
        # attack_client001_android_SD_ipad_video_scene01
        ##########################################################

        if kind == "attack":

            subject_id = parts[1].replace("client", "")
            subject_id = str(int(parts[1].replace("client", "")))

            return {
                "kind": "attack",
                "subject_id": subject_id,
                "camera_type": parts[2].lower(),
                "resolution": parts[3].lower(),
                "attack_type": "_".join(parts[4:-1]).lower(),
                "scene": parts[-1].lower(),
            }

        return None


    # def _read_txt_list(self, path: str) -> List[str]:
    #     with open(path, "r") as f:
    #         # return [line.strip() for line in f if line.strip()]
    #         return [
    #             line.strip().replace("client", "")
    #             for line in f
    #             if line.strip()
    #         ]

    def _read_txt_list(self, path: str) -> List[str]:
        with open(path, "r") as f:
            return [
                str(int(line.strip().replace("client", "")))
                for line in f
                if line.strip()
            ]

    # def _subject_split(self):
    #     train_ids = self._read_txt_list(self.train_list_path)
    #     test_ids = self._read_txt_list(self.test_list_path)

    #     if self.split == "test":
    #         return test_ids

    #     # deterministic train/val split from training subjects
    #     train_ids = sorted(train_ids)
    #     rng = random.Random(self.seed)
    #     rng.shuffle(train_ids)
    #     n_train = int(0.8 * len(train_ids))
    #     return train_ids[:n_train] if self.split == "train" else train_ids[n_train:]

    def _subject_split(self):

        train_ids = self._read_txt_list(self.train_list_path)
        test_ids = self._read_txt_list(self.test_list_path)

        ##########################################################
        # Official test
        ##########################################################

        if self.split == "test":
            return test_ids

        ##########################################################
        # No validation split
        ##########################################################

        if self.split == "val":
            return []

        ##########################################################
        # Official train
        ##########################################################

        if self.split == "train":
            return train_ids

        ##########################################################
        # Deterministic 80/20 split
        ##########################################################

        train_ids = sorted(train_ids)

        rng = random.Random(self.seed)

        rng.shuffle(train_ids)

        n_train = int(0.8 * len(train_ids))

        if self.split == "train80":
            return train_ids[:n_train]

        if self.split == "valtrain20":
            return train_ids[n_train:]

        raise ValueError(
            f"Unknown split: {self.split}"
        )


    def _protocol_match(self, meta: dict) -> bool:
        if meta is None:
            return False

        # Only use scene01
        if meta["scene"] != "scene01":
            return False

        # Subject split
        allowed_subjects = set(self._subject_split())
        if meta["subject_id"] not in allowed_subjects:
            return False

        # Grand protocol = all real + all attacks for the chosen split
        if self.protocol == "grand":
            return True

        if self.protocol not in self.PROTOCOL_MAP:
            raise ValueError(
                f"Unsupported MSU-MFSD protocol: {self.protocol}. "
                f"Use grand/all or one of {sorted(self.PROTOCOL_MAP.keys())}."
            )

        dest_camera, attack_type = self.PROTOCOL_MAP[self.protocol]

        # camera must match destination camera
        if meta["camera_type"] != dest_camera:
            return False

        # real video for that camera is always included
        if meta["kind"] == "real":
            return True

        # attack videos filtered by attack type
        return meta["attack_type"] == attack_type

    def _load_samples(self):
        samples = []
        allowed_subjects = set(self._subject_split())

        # Empty validation split
        if len(allowed_subjects) == 0:
            return []

        # print(f"\n >>>>>>>>>> [DEBUG] Allowed subjects ({len(allowed_subjects)}):")
        print(sorted(allowed_subjects)[:10])

        for cls_dir in [self.real_dir, self.attack_dir]:
            # print(f"\n >>>>>>>>>> [DEBUG] Scanning {cls_dir}")
            if not os.path.isdir(cls_dir):
                continue

            for root, _, files in os.walk(cls_dir):

                # print("\n >>>>>>>>>> [DEBUG] ROOT:", root)
                # print("\n >>>>>>>>>> [DEBUG] FILES:", len(files))

                for fname in files:
                    # print(f"\n >>>>>>>>>> [DEBUG] Checking file: {fname}")
                    if not fname.lower().endswith((".mov", ".mp4", ".avi")):
                        continue

                    vpath = os.path.join(root, fname)
                    stem = os.path.splitext(os.path.basename(fname))[0]
                    meta = self._parse_msu_stem(stem)

                    # print("\n >>>>>>>>>> [DEBUG] STEM:", stem)
                    # print("\n >>>>>>>>>> [DEBUG] META:", meta)

                    if meta is None:
                        continue

                    # Subject filtering
                    if meta["subject_id"] not in allowed_subjects:
                        # print(
                        #     "\n >>>>>>>>>> [DEBUG] SUBJECT ID:", meta["subject_id"],
                        #     "\n >>>>>>>>>> [DEBUG] IN ALLOWED SUBJECTS:", meta["subject_id"] in allowed_subjects,
                        # )
                        continue

                    # Grand protocol: accept everything for those subjects
                    if self.protocol == "grand":
                        pass
                    else:
                        if not self._protocol_match(meta):
                            continue

                    label = 1 if meta["kind"] == "real" else 0
                    # print(f'>>>>>>>>>>> [DEBUG]: {meta["subject_id"]}, {meta["kind"]}, {meta["camera_type"]}, {meta["attack_type"]}')
                    samples.append(
                        (
                            vpath,
                            label,
                            meta["subject_id"],
                            meta["camera_type"],
                            meta["attack_type"],
                        )
                    )

        # stable ordering
        return sorted(samples, key=lambda x: x[0])

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, video_path, frame_indices):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        for idx in frame_indices:
            if idx >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(F.to_tensor(frame))

        cap.release()

        # safe fallback if nothing could be read
        if not frames:
            h, w = self.target_size
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            frames.append(F.to_tensor(placeholder))

        while len(frames) < len(frame_indices):
            frames.append(frames[-1])

        return frames

    def __getitem__(self, idx):
        video_path, label, subject_id, camera_type, attack_type = self.samples[idx]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if self.is_train:
            start_frame = np.random.randint(0, max(1, total_frames - self.num_frames + 1))
        else:
            start_frame = 0

        frame_indices = np.linspace(
            start_frame,
            start_frame + self.num_frames - 1,
            self.num_frames,
            dtype=int
        )

        rgb_frames = self._load_frames(video_path, frame_indices)

        if self.transform:
            rgb_frames = [self.transform(frame) for frame in rgb_frames]

        if self.is_train:
            angle, scale = self._random_augmentation_params()
            rgb_frames = [self.apply_augmentation(frame, angle, scale) for frame in rgb_frames]

        rgb_frames = torch.stack(rgb_frames)

        if self.split == "test":
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            return {
                "img": rgb_frames,
                "label": label,
                "access_type": (video_id, camera_type, attack_type),
                "dataset": DOMAIN_IDS["MSU"]
            }

        return {
            "img": rgb_frames,
            "label": label,
            "dataset": DOMAIN_IDS["MSU"]
        }

    def _random_augmentation_params(self):
        angle = random.uniform(-180, 180) if random.random() > 0.5 else 0
        scale = random.uniform(0.7, 1.3) if random.random() > 0.5 else 1
        return angle, scale

    def apply_augmentation(self, image, angle, scale):
        if angle != 0:
            image = F.rotate(image, angle)
        if scale != 1:
            image = F.affine(image, angle=0, translate=(0, 0), scale=scale, shear=0)
        return image
    

# ==========================================================
# CASIA-FASD dataset class
# ==========================================================

class CASIA_FASD_Dataset(Dataset):
    """
    CASIA-FASD dataset loader.

    Directory structure:
        root/
            train/
                attack/
                real/
            test/
                attack/
                real/

    Labels:
        attack -> 0
        real   -> 1

    Split behavior:
        split="train"       -> all train dir data
        split="val"         -> empty dataset
        split="test"        -> all test dir data
        split="train80"     -> 80% of train dir data
        split="valtrain20"  -> remaining 20% of train dir data

    Notes:
        - train80 / valtrain20 are deterministic with a fixed seed.
        - train80 and valtrain20 never overlap.
    """

    def __init__(
        self,
        orig_root_dir: str,
        transform=None,
        num_frames: int = 16,
        is_train: bool = False,
        target_size: Tuple[int, int] = (224, 224),
        split: str = "train",
        seed: int = 1234,
    ):
        super().__init__()

        self.orig_root_dir = orig_root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.is_train = is_train
        self.target_size = target_size
        self.split = split.lower()
        self.seed = seed

        assert self.split in ("train", "val", "test", "train80", "valtrain20"), \
            "split must be one of: train, val, test, train80, valtrain20"

        self.train_dir = os.path.join(self.orig_root_dir, "train")
        self.test_dir = os.path.join(self.orig_root_dir, "test")

        self.samples = self._load_samples()

        print(f"CASIA-FASD Dataset (split={self.split}, is_train={self.is_train}) initialized!")
        print(f"Total samples: {len(self.samples)}")

    def _list_video_files(self, root_dir: str):
        samples = []
        if not os.path.isdir(root_dir):
            return samples

        for cls_name in ("attack", "real"):
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            for dirpath, _, filenames in os.walk(cls_dir):
                for fn in filenames:
                    if fn.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                        video_path = os.path.join(dirpath, fn)
                        label = 0 if cls_name == "attack" else 1
                        samples.append((video_path, label, cls_name))

        return sorted(samples, key=lambda x: x[0])

    def _fixed_split_train(self, samples):
        """
        Deterministic 80/20 split of train samples.
        """
        items = list(samples)
        rng = random.Random(self.seed)
        rng.shuffle(items)

        n_train = int(0.8 * len(items))
        train80 = items[:n_train]
        val20 = items[n_train:]
        return train80, val20

    def _load_samples(self):
        # test split: always test dir data
        if self.split == "test":
            return self._list_video_files(self.test_dir)

        # val split: no data
        if self.split == "val":
            return []

        # train split: all train dir data
        train_samples = self._list_video_files(self.train_dir)

        # 80/20 split of train dir data
        if self.split in ("train80", "valtrain20"):
            train80, val20 = self._fixed_split_train(train_samples)
            return train80 if self.split == "train80" else val20

        # default train split
        return train_samples

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, video_path, frame_indices):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        for idx in frame_indices:
            if idx >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(F.to_tensor(frame))

        cap.release()

        if not frames:
            h, w = self.target_size
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            frames.append(F.to_tensor(placeholder))

        while len(frames) < len(frame_indices):
            frames.append(frames[-1])

        return frames

    def __getitem__(self, idx):
        video_path, label, cls_name = self.samples[idx]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if self.is_train:
            start_frame = np.random.randint(0, max(1, total_frames - self.num_frames + 1))
        else:
            start_frame = 0

        frame_indices = np.linspace(
            start_frame,
            start_frame + self.num_frames - 1,
            self.num_frames,
            dtype=int
        )

        rgb_frames = self._load_frames(video_path, frame_indices)

        if self.transform:
            rgb_frames = [self.transform(frame) for frame in rgb_frames]

        if self.is_train:
            angle, scale = self._random_augmentation_params()
            rgb_frames = [self.apply_augmentation(frame, angle, scale) for frame in rgb_frames]

        rgb_frames = torch.stack(rgb_frames)

        if self.split == "test":
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            return {
                "img": rgb_frames,
                "label": label,
                "access_type": (video_id, cls_name),
                "dataset": DOMAIN_IDS["CASIA"]
            }
        return {
            "img": rgb_frames,
            "label": label,
            "dataset": DOMAIN_IDS["CASIA"]
        }

    def _random_augmentation_params(self):
        angle = random.uniform(-180, 180) if random.random() > 0.5 else 0
        scale = random.uniform(0.7, 1.3) if random.random() > 0.5 else 1
        return angle, scale

    def apply_augmentation(self, image, angle, scale):
        if angle != 0:
            image = F.rotate(image, angle)
        if scale != 1:
            image = F.affine(image, angle=0, translate=(0, 0), scale=scale, shear=0)
        return image


# ===============================
# Helper functions
# ===============================

def read_txt_list(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def split_train_val(items: List[str], is_train: bool, seed: int = 1234) -> List[str]:
    """
    Deterministic 80/20 split.
    """
    rng = random.Random(seed)
    items = sorted(items)
    rng.shuffle(items)
    n_train = int(0.8 * len(items))
    return items[:n_train] if is_train else items[n_train:]


def find_video_by_stem(root: str, cls: str, stem: str) -> Optional[str]:
    """
    Find video under root/cls/** whose basename (no extension) == stem.
    """
    cls_root = os.path.join(root, cls)
    for dirpath, _, filenames in os.walk(cls_root):
        for fn in filenames:
            if not fn.lower().endswith(('.mov', '.mp4', '.avi')):
                continue
            if os.path.splitext(fn)[0] == stem:
                return os.path.join(dirpath, fn)
    return None

####################################################################
################### Single dataset builder #########################
####################################################################

def build_single_dataset(dataset_name, split, args, transform):

    dataset_name = dataset_name.upper()

    ############################################################
    ######################## OULU ###############################
    ############################################################

    if dataset_name == "OULU":

        root = get_dataset_path("OULU")

        if split == "train":

            if args.oulu_protocol in ("3", "4"):
                protocol_flist = os.path.join(
                    root,
                    f"Baseline/Protocol_{args.oulu_protocol}/Train_{args.oulu_n_split}.txt"
                )
            else:
                protocol_flist = os.path.join(
                    root,
                    f"Baseline/Protocol_{args.oulu_protocol}/Train.txt"
                )

            return VideoDataset_Oulu(
                orig_root_dir=os.path.join(root, "Train_files"),
                file_list_path=protocol_flist,
                transform=transform,
                num_frames=args.num_frames,
                is_train=True,
                protocol=args.oulu_protocol,
            )

        elif split == "val":

            if args.oulu_protocol in ("3", "4"):
                protocol_flist = os.path.join(
                    root,
                    f"Baseline/Protocol_{args.oulu_protocol}/Dev_{args.oulu_n_split}.txt"
                )
            else:
                protocol_flist = os.path.join(
                    root,
                    f"Baseline/Protocol_{args.oulu_protocol}/Dev.txt"
                )

            return VideoDataset_Oulu(
                orig_root_dir=os.path.join(root, "Dev_files"),
                file_list_path=protocol_flist,
                transform=transform,
                num_frames=args.num_frames_val,
                is_train=False,
                protocol=args.oulu_protocol,
            )

        elif split == "test":

            if args.oulu_protocol in ("3", "4"):
                protocol_flist = os.path.join(
                    root,
                    f"Baseline/Protocol_{args.oulu_protocol}/Test_{args.oulu_n_split}.txt"
                )
            else:
                protocol_flist = os.path.join(
                    root,
                    f"Baseline/Protocol_{args.oulu_protocol}/Test.txt"
                )

            return VideoDataset_Oulu(
                orig_root_dir=os.path.join(root, "Test_files"),
                file_list_path=protocol_flist,
                transform=transform,
                num_frames=args.num_frames,
                is_train=False,
                split=split,
                protocol=args.oulu_protocol,
            )

    ############################################################
    ######################## SiW ################################
    ############################################################

    elif dataset_name == "SIW":

        root = get_dataset_path("SiW")

        return SIW_Dataset(
            orig_root_dir=root,
            transform=transform,
            num_frames=args.num_frames if split != "val" else args.num_frames_val,
            is_train=(split == "train"),
            protocol=args.siw_protocol,
            split=split,
        )

    ############################################################
    ######################## MSU ################################
    ############################################################

    # elif dataset_name in ("MSU", "MSU-MFSD", "MSU_MFSD"):

    #     root = get_dataset_path("MSU")

    #     msu_split = split

    #     if (
    #         split == "val"
    #         and args.training_mode == "loo"
    #         and args.loo_val_source == "leave_out"
    #         and dataset_name == args.leave_out.upper()
    #     ):
    #         msu_split = "train"

    #     return MSU_MFSD_Dataset(
    #         orig_root_dir=root,
    #         transform=transform,
    #         num_frames=args.num_frames if split != "val" else args.num_frames_val,
    #         is_train=(msu_split == "train"),
    #         protocol=args.msu_protocol,
    #         split=msu_split,
    #     )

    elif dataset_name in ("MSU", "MSU-MFSD", "MSU_MFSD"):

        root = get_dataset_path("MSU")

        if split == "train":

            msu_split = args.msu_split

        elif split == "val":

            # LOO validation comes from leave-out dataset
            if (
                args.training_mode == "loo"
                and args.loo_val_source == "leave_out"
                and dataset_name.upper() == args.leave_out.upper()
            ):

                # MSU has no official validation split
                if args.msu_split == "train":
                    msu_split = "train"
                else:
                    msu_split = "valtrain20"

            else:
                if args.msu_split == "train":
                    msu_split = "val"
                else:
                    msu_split = "valtrain20"

        else:

            msu_split = "test"

        return MSU_MFSD_Dataset(
            orig_root_dir=root,
            transform=transform,
            num_frames=args.num_frames if split != "val" else args.num_frames_val,
            is_train=(msu_split in ("train", "train80")),
            protocol=args.msu_protocol,
            split=msu_split,
        )

    ############################################################
    ######################## CASIA ##############################
    ############################################################

    elif dataset_name in ("CASIA", "CASIA-FASD", "CASIA_FASD"):

        root = get_dataset_path("CASIA")

        if split == "train":
            casia_split = args.casia_split

        elif split == "val":
            # LOO validation comes from leave-out dataset
            if (
                args.training_mode == "loo"
                and args.loo_val_source == "leave_out"
                and dataset_name.upper() == args.leave_out.upper()
            ):

                # CASIA has no official validation split
                if args.casia_split == "train":
                    casia_split = "train"
                else:
                    casia_split = "valtrain20"

            else:
                if args.casia_split == "train":
                    casia_split = "val"
                else:
                    casia_split = "valtrain20"

        else:
            casia_split = "test"

        return CASIA_FASD_Dataset(
            orig_root_dir=root,
            transform=transform,
            num_frames=args.num_frames if split != "val" else args.num_frames_val,
            is_train=(split == "train"),
            split=casia_split,
        )

    ############################################################
    ################ ReplayAttack / Rose ########################
    ############################################################

    else:

        root = get_dataset_path(dataset_name)

        if split == "train":
            folder = "train"
            is_train = True
            nframes = args.num_frames

        elif split == "val":
            folder = "devel"
            is_train = False
            nframes = args.num_frames_val

        else:
            folder = "test"
            is_train = False
            nframes = args.num_frames

        return VideoDataset(
            orig_root_dir=os.path.join(root, folder),
            transform=transform,
            num_frames=nframes,
            is_train=is_train,
            datasetname=dataset_name,
        )



def build_train_val_datasets(args, train_transform, val_transform,):

    if args.training_mode == "single":

        train_dataset = build_single_dataset(
            args.datasets[0],
            "train",
            args,
            train_transform,
        )

        val_dataset = build_single_dataset(
            args.datasets[0],
            "val",
            args,
            val_transform,
        )

        return train_dataset, val_dataset

    train_sets = []
    val_sets = []

    if args.training_mode == "joint":

        datasets = args.datasets

    elif args.training_mode == "loo":
        # datasets = [ds for ds in args.datasets if ds != args.leave_out]

        datasets = [ds for ds in args.datasets if ds != args.leave_out]

        # Train set is always the concatenation of the remaining datasets
        for ds in datasets:
            train_sets.append(
                build_single_dataset(
                    ds,
                    "train",
                    args,
                    train_transform,
                )
            )

        # Validation source
        if args.loo_val_source == "leave_out":
            val_dataset = build_single_dataset(
                args.leave_out,
                "val",
                args,
                val_transform,
            )
        else:
            val_sets = []
            for ds in datasets:
                val_sets.append(
                    build_single_dataset(
                        ds,
                        "val",
                        args,
                        val_transform,
                    )
                )
            val_dataset = ConcatDataset(val_sets)

        return (
            ConcatDataset(train_sets),
            val_dataset,
        )

    else:
        raise ValueError(
            f"Unknown training mode {args.training_mode}"
        )

    for ds in datasets:

        train_sets.append(
            build_single_dataset(
                ds,
                "train",
                args,
                train_transform,
            )
        )

        val_sets.append(
            build_single_dataset(
                ds,
                "val",
                args,
                val_transform,
            )
        )

    return (
        ConcatDataset(train_sets),
        ConcatDataset(val_sets),
    )


def build_test_dataset(args, transform):

    # if args.training_mode == "single":
    #     dataset = args.datasets[0]

    # elif args.training_mode == "joint":
    #     if args.dataset is None:
    #         raise ValueError(
    #             "Joint evaluation requires --dataset."
    #         )
    #     dataset = args.dataset

    # elif args.training_mode == "loo":
    #     dataset = args.leave_out
    #     if args.dataset is not None:
    #         dataset = args.dataset


    if getattr(args, "dataset", None) is not None:
        dataset = args.dataset

    elif args.training_mode == "single":
        dataset = args.datasets[0]

    elif args.training_mode == "joint":
        dataset = args.datasets[0]

    elif args.training_mode == "loo":
        dataset = args.leave_out

    else:
        raise ValueError(f"Unknown training mode {args.training_mode}")

    return build_single_dataset(
        dataset,
        "test",
        args,
        transform,
    )

