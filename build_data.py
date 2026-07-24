import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import random
from typing import Tuple, List, Optional


##########################################################
####### Dataset class for OULU-NPY dataset #######

class VideoDataset_Oulu(Dataset):
    def __init__(self, orig_root_dir, file_list_path, transform=None, num_frames=16, is_train=False, protocol=None):
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

        if self.is_train:
            return {'img':combined_frames, 'label':label}
        else:
            return {'img':combined_frames, 'label':label, 'access_type':access_type}

        
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
    def __init__(self, orig_root_dir, transform=None, num_frames=16, is_train=False):
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

        return {'img':combined_frames, 'label':label}
        
        
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

######################################################
# ------------------------------------
# Dataset builders (train/val and test)
# ------------------------------------

def build_train_val_datasets(args, transform):
    
    if args.dataset=='OULU':

        if args.protocol=='3' or args.protocol=='4':
            protocol_flist = os.path.join(args.orig_dataset_path, f'Baseline/Protocol_{args.protocol}/Train_{args.n_split}.txt')
        else:
            protocol_flist = os.path.join(args.orig_dataset_path, f'Baseline/Protocol_{args.protocol}/Train.txt')
        train_dataset = VideoDataset_Oulu(
            orig_root_dir=args.orig_dataset_path+'/Train_files',
            file_list_path = protocol_flist,
            transform=transform,
            num_frames=args.num_frames,
            is_train=True,
            protocol = args.protocol
        )

        if args.protocol=='3' or args.protocol=='4':
            protocol_flist = os.path.join(args.orig_dataset_path, f'Baseline/Protocol_{args.protocol}/Dev_{args.n_split}.txt')
        else:
            protocol_flist = os.path.join(args.orig_dataset_path, f'Baseline/Protocol_{args.protocol}/Dev.txt')
        val_dataset = VideoDataset_Oulu(
            orig_root_dir=args.orig_dataset_path+'/Dev_files',
            file_list_path = protocol_flist,
            transform=transform,
            num_frames=args.num_frames_val,
            is_train=False,
            protocol = args.protocol
        )

    elif args.dataset=='SiW':
        train_dataset = SIW_Dataset(
            orig_root_dir=args.orig_dataset_path,
            transform=transform,
            num_frames=args.num_frames,
            is_train=True,
            protocol = args.protocol,
            split = "train"
        )
        val_dataset = SIW_Dataset(
            orig_root_dir=args.orig_dataset_path,
            transform=transform,
            num_frames=args.num_frames_val,
            is_train=False,
            protocol = args.protocol,
            split = "val"
        )

    else:
        train_dataset = VideoDataset(
            orig_root_dir=args.orig_dataset_path+'/train',
            transform=transform,
            num_frames=args.num_frames,
            is_train=True,
        )
        val_dataset = VideoDataset(
            orig_root_dir=args.orig_dataset_path+'/devel',
            transform=transform,
            num_frames=args.num_frames_val,
            is_train=False,
        )

    return train_dataset, val_dataset


def build_test_dataset(args, transform):

    if args.dataset=='OULU':
        # print(f'[DEBUG] OULU Test protocol-file: args.orig_dataset_path {args.orig_dataset_path}, ')
        if args.protocol=='3' or args.protocol=='4':
            protocol_flist = os.path.join(args.orig_dataset_path, f'Baseline/Protocol_{args.protocol}/Test_{args.n_split}.txt')
        else:
            protocol_flist = os.path.join(args.orig_dataset_path, f'Baseline/Protocol_{args.protocol}/Test.txt')
        test_dataset = VideoDataset_Oulu(
            orig_root_dir=args.orig_dataset_path+'/Test_files',
            file_list_path = protocol_flist,
            transform=transform,
            num_frames=args.num_frames,
            is_train=False,
            protocol = args.protocol
        )
    elif args.dataset=='SiW':
        test_dataset = SIW_Dataset(
            orig_root_dir=args.orig_dataset_path,
            transform=transform,
            num_frames=args.num_frames,
            is_train=False,
            protocol = args.protocol,
            split = "test"
        )
    else:
        test_dataset = VideoDataset(
            orig_root_dir=args.orig_dataset_path+'/test',
            transform=transform,
            num_frames=args.num_frames,
            is_train=False,
        )

    return test_dataset
