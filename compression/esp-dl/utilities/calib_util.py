import time
from typing import Callable

import onnxruntime
import pandas as pd
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from esp_ppq.executor.torch import TorchExecutor
from esp_ppq.IR import BaseGraph
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def load_pv_from_directory(
    directory: str,
    subset: int = None,
    batchsize: int = 32,
    shuffle: bool = False,
    require_label: bool = True,
    num_of_workers: int = 12,
    img_height: int = 224,
    img_width: int = 224
) -> tuple[datasets.ImageFolder, torch.utils.data.DataLoader]:
    """
    A standardized Imagenet data loading process,
    directory: The location where the data is loaded
    subset: If set to a non-empty value, a subset of the specified size is extracted from the dataset
    batchsize: The batch size of the data loader
    require_label: Whether labels are required
    shuffle: Whether to shuffle the dataset
    """
    dataset = datasets.ImageFolder(
        directory,
        transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: (x - 0.5) * 2),  # Rescale from [0, 1] to [-1, 1]
            # transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Change from (C, H, W) to (H, W, C)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to [-1, 1]
        ]),
    )

    if subset:
        dataset = Subset(dataset, indices=[_ for _ in range(0, subset)])
    if require_label:
        dataloader = torch.utils.data.DataLoader(
            dataset = dataset,
            batch_size = batchsize,
            shuffle = shuffle,
            num_workers = num_of_workers,
            pin_memory = False,
            drop_last = True,  # The onnx model does not support dynamic batch size, the last batch may be misaligned, so drop it
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset = dataset,
            batch_size = batchsize,
            shuffle = shuffle,
            num_workers = num_of_workers,
            pin_memory = False,
            collate_fn = lambda x: torch.cat([sample[0].unsqueeze(0) for sample in x], dim=0),
            drop_last = False,  # Data without labels is calibration data, so no need to drop
        )
    return dataset, dataloader


def evaluate_torch_module_with_imagenet(
    model: torch.nn.Module,
    imagenet_validation_dir: str = None,
    batchsize: int = 32,
    device: str = "cuda",
    imagenet_validation_loader: DataLoader = None,
    verbose: bool = True,
    img_height: int = 224,
    img_width: int = 224,
    print_confusion_matrix: bool = False,
    confusion_matrix_path: str = "confusion_matrix.png",
    topk: tuple = (1,),
) -> pd.DataFrame:
    
    model.eval()
    with torch.no_grad():
        model_forward_function = lambda input_tensor: model(input_tensor)
        return _evaluate_any_module_with_imagenet(
            model_forward_function=model_forward_function,
            batchsize=batchsize,
            device=device,
            imagenet_validation_dir=imagenet_validation_dir,
            imagenet_validation_loader=imagenet_validation_loader,
            verbose=verbose,
            img_height=img_height,
            img_width=img_width,
            print_confusion_matrix=print_confusion_matrix,
            confusion_matrix_path=confusion_matrix_path,
            topk=topk
        )


def evaluate_ppq_module_with_pv(
    model: BaseGraph,
    imagenet_validation_dir: str = None,
    batchsize: int = 32,
    device: str = "cuda",
    imagenet_validation_loader: DataLoader = None,
    verbose: bool = True,
    img_height: int = 224,
    img_width: int = 224,
    print_confusion_matrix: bool = False,
    confusion_matrix_path: str = "confusion_matrix.png",
    topk: tuple = (1,)
) -> pd.DataFrame:
    """
    A logic set for testing PPQ modules,
    simply feed in ppq.IR.BaseGraph (approx translation)
    """
    executor = TorchExecutor(graph=model, device=device)
    model_forward_function = lambda input_tensor: torch.tensor(executor(*[input_tensor])[0])
    return _evaluate_any_module_with_pv(
        model_forward_function=model_forward_function,
        batchsize=batchsize,
        device=device,
        imagenet_validation_dir=imagenet_validation_dir,
        imagenet_validation_loader=imagenet_validation_loader,
        verbose=verbose,
        img_height=img_height,
        img_width=img_width,
        print_confusion_matrix=print_confusion_matrix,
        confusion_matrix_path=confusion_matrix_path,
        topk=topk,
    )


def _evaluate_any_module_with_pv(
    model_forward_function: Callable,
    imagenet_validation_dir: str,
    batchsize: int = 32,
    topk: tuple = (1,),
    device: str = "cuda",
    imagenet_validation_loader: DataLoader = None,
    verbose: bool = True,
    img_height: int = 224,
    img_width: int = 224,
    print_confusion_matrix: bool = False,
    confusion_matrix_path: str = "confusion_matrix.png"
):
    """
    A very standard ImageNet testing logic (approx translation)
    """
    recorder = {"loss": [], "batch_time": []}
    for k in topk:
        recorder[f"top{k}_accuracy"] = []
    all_preds = []
    all_labels = []

    if imagenet_validation_loader is None:
        imagenet_validation_loader = load_pv_from_directory(
            imagenet_validation_dir,
            batchsize=batchsize,
            shuffle=False,
            img_width=img_width,
            img_height=img_height,
        )

    loss_fn = torch.nn.CrossEntropyLoss().to("cpu")

    for batch_idx, (batch_input, batch_label) in tqdm(enumerate(imagenet_validation_loader), desc="Evaluating Model...", total=len(imagenet_validation_loader)):
        batch_input = batch_input.to(device)
        batch_label = batch_label.to(device)
        batch_time_mark_point = time.time()

        batch_pred = model_forward_function(batch_input)
        if isinstance(batch_pred, list):
            batch_pred = torch.tensor(batch_pred)

        recorder["batch_time"].append(time.time() - batch_time_mark_point)
        recorder["loss"].append(loss_fn(batch_pred.to("cpu"), batch_label.to("cpu")))

        precs = accuracy(torch.tensor(batch_pred).to("cpu"), batch_label.to("cpu"), topk=topk)
        for i, k in enumerate(topk):
            recorder[f"top{k}_accuracy"].append(precs[i].item())

        # Store predictions and labels for confusion matrix
        all_preds.extend(torch.argmax(batch_pred, dim=1).cpu().numpy())
        all_labels.extend(batch_label.cpu().numpy())

        if batch_idx % 100 == 0 and verbose:
            topk_status = "\t".join([
                f"Prec@{k} {sum(recorder[f'top{k}_accuracy']) / len(recorder[f'top{k}_accuracy']):.3f} ({sum(recorder[f'top{k}_accuracy']) / len(recorder[f'top{k}_accuracy']):.3f})"
                for k in topk
            ])
            print(f"Test: [{batch_idx} / {len(imagenet_validation_loader)}]\t{topk_status}")

    if verbose:
        topk_status = " ".join([
            f"Prec@{k} {sum(recorder[f'top{k}_accuracy']) / len(recorder[f'top{k}_accuracy']):.3f}"
            for k in topk
        ])
        print(f" * {topk_status}")

    if print_confusion_matrix:
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(cm)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(confusion_matrix_path)

    # dump records toward dataframe
    dataframe = pd.DataFrame()
    for column_name in recorder:
        dataframe[column_name] = recorder[column_name]
    return dataframe


def _evaluate_any_module_with_imagenet(
    model_forward_function: Callable,
    imagenet_validation_dir: str,
    batchsize: int = 32,
    topk: tuple = (1,),
    device: str = "cuda",
    imagenet_validation_loader: DataLoader = None,
    verbose: bool = True,
    img_height: int = 224,
    img_width: int = 224,
    print_confusion_matrix: bool = False,
    confusion_matrix_path: str = "confusion_matrix.png",
):
    """
    A very standard ImageNet testing logic (approx translation)
    """
    recorder = {"loss": [], "batch_time": []}
    for k in topk:
        recorder[f"top{k}_accuracy"] = []
    all_preds = []
    all_labels = []

    if imagenet_validation_loader is None:
        imagenet_validation_loader = load_imagenet_from_directory(
            imagenet_validation_dir,
            batchsize=batchsize,
            shuffle=False,
            img_width=img_width,
            img_height=img_height,
        )

    loss_fn = torch.nn.CrossEntropyLoss().to("cpu")

    for batch_idx, (batch_input, batch_label) in tqdm(
        enumerate(imagenet_validation_loader),
        desc="Evaluating Model...",
        total=len(imagenet_validation_loader),
    ):
        batch_input = batch_input.to(device)
        batch_label = batch_label.to(device)
        batch_time_mark_point = time.time()

        batch_pred = model_forward_function(batch_input)
        if isinstance(batch_pred, list):
            batch_pred = torch.tensor(batch_pred)

        recorder["batch_time"].append(time.time() - batch_time_mark_point)
        recorder["loss"].append(loss_fn(batch_pred.to("cpu"), batch_label.to("cpu")))

        precs = accuracy(torch.tensor(batch_pred).to("cpu"), batch_label.to("cpu"), topk=topk)
        for i, k in enumerate(topk):
            recorder[f"top{k}_accuracy"].append(precs[i].item())

        # Store predictions and labels for confusion matrix
        all_preds.extend(torch.argmax(batch_pred, dim=1).cpu().numpy())
        all_labels.extend(batch_label.cpu().numpy())

        if batch_idx % 100 == 0 and verbose:
            topk_status = "\t".join([f"Prec@{k} {sum(recorder[f'top{k}_accuracy']) / len(recorder[f'top{k}_accuracy']):.3f} ({sum(recorder[f'top{k}_accuracy']) / len(recorder[f'top{k}_accuracy']):.3f})" for k in topk])
            print(f"Test: [{batch_idx} / {len(imagenet_validation_loader)}]\t{topk_status}")

    if verbose:
        topk_status = " ".join([f"Prec@{k} {sum(recorder[f'top{k}_accuracy']) / len(recorder[f'top{k}_accuracy']):.3f}" 
        for k in topk])
        print(f" * {topk_status}")

    if print_confusion_matrix:
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(cm)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(confusion_matrix_path)

    dataframe = pd.DataFrame()
    for column_name in recorder:
        dataframe[column_name] = recorder[column_name]
    return dataframe


def load_imagenet_from_directory(
    directory: str,
    subset: int = None,
    batchsize: int = 32,
    shuffle: bool = False,
    require_label: bool = True,
    num_of_workers: int = 12,
    img_height: int = 224,
    img_width: int = 224
) -> torch.utils.data.DataLoader:
    """
    A standardized Imagenet data loading process,
    directory: The location where the data is loaded
    subset: If set to a non-empty value, a subset of the specified size is extracted from the dataset
    batchsize: The batch size of the data loader
    require_label: Whether labels are required
    shuffle: Whether to shuffle the dataset
    """
    
    dataset = datasets.ImageFolder(
        directory,
        transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize for MobileNet
        ])
    )

    if subset:
        dataset = Subset(dataset, indices=[_ for _ in range(0, subset)])
    if require_label:
        return torch.utils.data.DataLoader(
            dataset = dataset,
            batch_size = batchsize,
            shuffle = shuffle,
            num_workers = num_of_workers,
            pin_memory = False,
            drop_last = True,  # The onnx model does not support dynamic batch size, the last batch may be misaligned, so drop it
        )
    else:
        return torch.utils.data.DataLoader(
            dataset = dataset,
            batch_size = batchsize,
            shuffle = shuffle,
            num_workers = num_of_workers,
            pin_memory = False,
            collate_fn = lambda x: torch.cat([sample[0].unsqueeze(0) for sample in x], dim=0),
            drop_last = False,  # Data without labels is calibration data, so no need to drop
        )
