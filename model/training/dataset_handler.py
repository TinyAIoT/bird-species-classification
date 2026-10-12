import os
from collections import Counter

import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import Lambda as L
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from utils.training_utils import get_default_device, check_corrupted, to_device
from utils.dataset_utils import SaveEvery_nth_image, CenterCrop, Image_Name_Saver, ImageFolderWithFilename, MappingConcatDataset

# Transforms with randomness, they belong in `train_transforms` only
RANDOM_TRANSFORMS = {"RandomResizedCrop", "RandomCrop", "RandomHorizontalFlip", "ColorJitter"}


class DatasetHandler:
    def __init__(self, config):
        """Initialize the DatasetHandler with configuration parameters
        Args:
            config (dict): Dictionary containing configuration parameters
        """
        self.batch_size = config["batch_size"]
        # self.data_path = config["dataset_path"]
        self.num_workers = config["num_workers"]
        self.check_corrupted_images = config.get("check_corrupted_images", False)
        self.device = get_default_device()
        self.pin_memory = config.get("pin_memory", True)  # Pin memory for faster data transfer to GPU, not sure if this is needed for MPS and CPU
        self.datasets = config.get("datasets", [])
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.class_names = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def build_transforms(self):
        """Build per dataset one transform for training and one for evaluation (val/test).

        `transforms` are used for val/test, and also for training if no `train_transforms` are given.
        `train_transforms` (optional) are only used for training, e.g. to add random augmentations.
        """
        self.transforms_built = {}
        for dataset in self.datasets:
            eval_list = dataset.get("transforms", [])
            random_in_eval = [k for t in eval_list if isinstance(t, dict) for k in t if k in RANDOM_TRANSFORMS]
            if random_in_eval:
                print(f"Warning: dataset {dataset['name']} has random transforms {random_in_eval} in `transforms`, "
                      f"they are also applied to val/test. Move them to `train_transforms`.")
            eval_transform = self._compose(eval_list)
            train_list = dataset.get("train_transforms")
            train_transform = self._compose(train_list) if train_list is not None else eval_transform
            self.transforms_built[dataset["name"]] = (train_transform, eval_transform)

        print("Built the following transforms for datasets: \n")
        for name, (train_transform, eval_transform) in self.transforms_built.items():
            print(f"Dataset: {name}\n  Train transform: {train_transform}\n  Val/test transform: {eval_transform} \n")
        return eval_transform

    def _compose(self, transform_list):
        """Build a torchvision Compose from a list of transform configs (see training_config_template.yaml)."""
        dataset_transforms = []
        name_saver = Image_Name_Saver()
        dataset_transforms.append(name_saver)
        for t in transform_list:
            if "resize" in t:
                params = t["resize"]
                height = params["height"]
                width = params["width"]
                dataset_transforms.append(transforms.Resize((height, width)))
            if "RandomResizedCrop" in t:
                params = t["RandomResizedCrop"]                    
                height = params["height"]
                width = params["width"]
                scale = params["scale"]
                dataset_transforms.append(transforms.RandomResizedCrop((height, width), scale=scale))
            if "RandomCrop" in t:
                params = t["RandomCrop"]
                height = params["height"]
                width = params["width"]
                padding = params.get("padding", 0)
                pad_if_needed = params.get("pad_if_needed", False)
                fill = params.get("fill", 0)
                padding_mode = params.get("padding_mode", "constant")
                dataset_transforms.append(transforms.RandomCrop((height, width), padding=padding, pad_if_needed=pad_if_needed, 
                                                                fill=fill, padding_mode=padding_mode))
            if "rotate" in t:
                degrees = t["rotate"].get("degrees", 0)
                dataset_transforms.append(L(lambda img: F.rotate(img, degrees)))
            if "RandomHorizontalFlip" in t:
                params = t["RandomHorizontalFlip"]
                p = params["probability"]
                dataset_transforms.append(transforms.RandomHorizontalFlip(p=p))
            if "crop_relative" in t:
                params = t["crop_relative"]
                top = params.get("top", 0.0)
                left = params.get("left", 0.0)
                height = params["height"]
                width = params["width"]
                dataset_transforms.append(L(lambda img: F.resized_crop(img, int(top * img.height), int(left * img.width), 
                                                                       int(height * img.height), int(width * img.width), (img.height, img.width))))
            if "CenterCrop" in t:
                params = t["CenterCrop"]
                side = params.get("side", "center")
                height = params["height"]
                width = params["width"]
                dataset_transforms.append(CenterCrop(side, height, width))
            if "crop" in t:
                params = t["crop"]
                top = params.get("top", 0)
                left = params.get("left", 0)
                height = params["height"]
                width = params["width"]
                dataset_transforms.append(L(lambda img: F.crop(img, top, left, height, width)))
            if "ColorJitter" in t:
                params = t["ColorJitter"]
                b = params["brightness"]
                c = params["contrast"]
                s = params["saturation"]
                h = params["hue"]
                dataset_transforms.append(transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h))
            if "save_img" in t:
                params = t["save_img"]
                n = params.get("n", 100)
                save_dir = params.get("save_dir", "debug_images")
                dataset_transforms.append(SaveEvery_nth_image(n=n, save_dir=save_dir, name_saver=name_saver))
            if "ToTensor" in t:
                dataset_transforms.append(transforms.ToTensor())
            if "normalize" in t:
                params = t["normalize"]
                mean = params.get("mean", [0.485, 0.456, 0.406])
                std = params.get("std", [0.229, 0.224, 0.225])
                dataset_transforms.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(dataset_transforms)

    def load_data(self):
        """
        Load (multiple) datasets and apply transformations
        """
        train_paths = []
        val_paths = []
        test_paths = []
        self.build_transforms()
        for ds in self.datasets:
            train_path = os.path.join(ds["path"], "train")
            val_path = os.path.join(ds["path"], "val")
            test_path = os.path.join(ds["path"], "test")
            ds["train_path"] = train_path
            ds["val_path"] = val_path
            ds["test_path"] = test_path
            train_paths.append(train_path)
            val_paths.append(val_path)
            test_paths.append(test_path)

        for path in train_paths + val_paths + test_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Dataset path {path} does not exist.")
            
        train_datasets = []
        val_datasets = []
        test_datasets = []
        for ds in self.datasets:
            name = ds["name"]
            train_path = ds["train_path"]
            val_path = ds["val_path"]
            test_path = ds["test_path"]
            train_transform, eval_transform = self.transforms_built[name]
            train_datasets.append(ImageFolderWithFilename(train_path, train_transform))
            val_datasets.append(ImageFolderWithFilename(val_path, eval_transform))
            test_datasets.append(ImageFolderWithFilename(test_path, eval_transform))

        self.train_data = ConcatDataset(train_datasets)
        self.val_data = ConcatDataset(val_datasets)
        self.test_data = ConcatDataset(test_datasets)
        if self.check_corrupted_images:
            # Check for corrupted images by trying to open and fully load each file
            for ds in self.train_data.datasets:
                check_corrupted(ds.samples, "train")
            for ds in self.val_data.datasets:
                check_corrupted(ds.samples, "val")
            for ds in self.test_data.datasets:
                check_corrupted(ds.samples, "test")

        all_classes = set()
        for ds in self.train_data.datasets:
            print(f"Classes in training dataset {ds.root}: \n {ds.class_to_idx} \n")
            all_classes.update(ds.classes)
            if all_classes != set(ds.classes):
                print(f"Warning: Datasets contain different classes. Check for correct index mapping")
            
        self.class_names = sorted(list(all_classes))
        print("All class names: \n")
        print(self.class_names)
        print("Train dataset size:", len(self.train_data))
        print("Validation dataset size:", len(self.val_data))
        print("Test dataset size:", len(self.test_data))

    def preprocess_data(self):
        """Create DataLoaders for training, validation, and testing"""
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        self.val_loader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
        self.test_loader = DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def get_train_class_counts(self):
        """Return the number of training images per class, in the order of get_class_names()
        Returns:
            torch.Tensor: Image counts per class
        """
        counts = torch.zeros(len(self.class_names))
        for ds in self.train_data.datasets:
            for target, n in Counter(ds.targets).items():
                counts[self.class_names.index(ds.classes[target])] += n
        return counts

    def get_class_names(self):
        """Return the class names
        Returns:
            list: List of class names in the dataset
        """
        return self.class_names

    def get_data_loaders(self):
        """Return all data loaders
        Returns:
            tuple: Tuple containing train, validation, and test DataLoaders
        """
        # breakpoint()  # Debugging point to check if data loaders are created
        # print structure of tensor within data loaders
        if (self.train_loader is None or self.val_loader is None or self.test_loader is None):
            raise ValueError("Data loaders have not been created. Call preprocess_data() first.")
        # Return the data loaders
        return self.train_loader, self.val_loader, self.test_loader

    def print_data_loader_structure(self, loader):
        """Return all data loaders
        Returns:
            tuple: Tuple containing train, validation, and test DataLoaders
        """
        # breakpoint()  # Debugging point to check if data loaders are created
        # print structure of tensor within data loaders
        if loader is None:
            raise ValueError("Data loader has not been created yet. Call preprocess_data() first.")
        for batch in loader:
            print(batch[0].shape, batch[1].shape)
            break

    def get_device(self):
        """Return the device being used
        Returns:
            torch.device: Device used for training (CPU, CUDA, or MPS)
        """
        return self.device
