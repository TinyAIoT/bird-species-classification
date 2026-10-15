# Description: This script quantizes a pre-trained PyTorch model using the ESP-DL library.
# The script reads the configuration file and the model file, quantizes the model, and evaluates the quantized model.
# The script saves the quantized model and the evaluation results.

# The config file should contain the following fields:
# - batch_size: The batch size for the calibration and testing datasets.
# - input_model_path: The path to the pre-trained PyTorch model file (full pickled model, *.pth).
# - dataset_path: The path to the dataset directory. The directory should contain the train (calibration) and test splits.
# - output_path: The path to the output directory.
# - model_name: The name of the model.
# - quant_config: The quantization configuration. The supported values are None, "LayerwiseEqualization_quantization", and "MixedPrecision_quantization".
# Optional fields:
# - mixed_precision_layers: Names of the layers quantized with 16 bit (required for "MixedPrecision_quantization").
# - calib_size: Number of train images used for calibration, balanced over the classes (default: 1024).
# - img_height, img_width: Model input size (default: 224).
# - topk: Top-k accuracies to report, e.g. [1, 3] (default: 1).
# - num_workers: DataLoader workers (default: 4).
# - seed: Random seed for the calibration subset (default: 0).

# -------------------------------------------
# Import Libraries
# --------------------------------------------
import os
from typing import List, Optional
import yaml
import json
import math
import torch
from utilities.calib_util import (
    load_pv_from_directory,
    evaluate_ppq_module_with_pv,
    evaluate_torch_module_with_imagenet,
)
from esp_ppq import QuantizationSettingFactory, QuantizationSetting
from esp_ppq.api import espdl_quantize_torch, get_target_platform
import argparse
import pandas as pd

QUANT_METHODS = ["None", "LayerwiseEqualization_quantization", "MixedPrecision_quantization"]

# -------------------------------------------
# Helper Functions
# --------------------------------------------


# Pretty print the quantization settings
def pretty_print_settings(d, indent=0):
    for key, value in d.items():
        print("\t" * indent + str(key))
        if hasattr(value, "__dict__"):
            if isinstance(value.__dict__, dict):
                pretty_print_settings(value.__dict__, indent + 1)
        else:
            print("\t" * (indent + 1) + str(value))


def parse_quant_method(quant_config) -> str:
    """Normalize the quant_config entry of the config (YAML null, "None" or a method name) to one of QUANT_METHODS."""
    method = "None" if quant_config is None else str(quant_config).strip()
    if method not in QUANT_METHODS:
        raise ValueError(f"Unsupported quant_config '{quant_config}'. Supported: {QUANT_METHODS}")
    return method


# Set the quantization settings for the model
def set_quant_settings(method: str, args, target: str, mixed_precision_layers: Optional[List[str]] = None) -> QuantizationSetting:
    """Create the ESP-DL quantization setting for the given method.

    Args:
        method (str): one of QUANT_METHODS
        -'MixedPrecision_quantization': if some layers in model have larger errors in 8-bit quantization, dispatching
                                        the layers to 16-bit quantization. The layers are given by mixed_precision_layers
                                        (operation names of the exported graph, e.g. "/features/features.0/Conv").
        -'LayerwiseEqualization_quantization': using weight equalization strategy, which is proposed by Markus Nagel.
                                                Refer to paper https://openaccess.thecvf.com/content_ICCV_2019/papers/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.pdf for more information.
                                                If ReLu6 layers are used in model, make sure to convert ReLU6 to ReLU for better precision.
        args: parsed command line arguments with the equalization parameters
        target (str): ESP-DL target platform, e.g. "esp32s3"
        mixed_precision_layers (List[str]): layers quantized with 16 bit for 'MixedPrecision_quantization'
    Returns:
        QuantizationSetting
    """
    quant_setting = QuantizationSettingFactory.espdl_setting()
    if method == "MixedPrecision_quantization":
        if not mixed_precision_layers:
            raise ValueError("MixedPrecision_quantization needs the layer names to quantize with 16 bit in "
                             "'mixed_precision_layers' of the config (names depend on the architecture).")
        for layer in mixed_precision_layers:
            quant_setting.dispatching_table.append(layer, get_target_platform(target, 16, True))
    elif method == "LayerwiseEqualization_quantization":
        quant_setting.equalization = True
        quant_setting.equalization_setting.iterations = args.iterations
        quant_setting.equalization_setting.value_threshold = args.value_threshold
        quant_setting.equalization_setting.opt_level = args.opt_level
        quant_setting.equalization_setting.including_bias = args.including_bias
        quant_setting.equalization_setting.bias_multiplier = args.bias_multiplier
        quant_setting.equalization_setting.including_act = args.including_act
        quant_setting.equalization_setting.act_multiplier = args.act_multiplier
        quant_setting.equalization_setting.interested_layers = None

    print("Quantization settings: ")
    pretty_print_settings(quant_setting.__dict__)
    return quant_setting


def write_class_names_header(class_names: List[str], path: str):
    """Write the class names in model output order as C++ header for the firmware
    (deployment/main/include/classification_category_name.hpp)."""
    with open(path, "w") as f:
        f.write("#pragma once\n")
        f.write(f"// Generated by quantize_torch_model.py, {len(class_names)} classes in model output order\n")
        f.write("static const char *classification_cat_names[] = {\n")
        f.writelines(f'    "{name}",\n' for name in class_names)
        f.write("};\n")


def make_collate_fn(device: str):
    def collate_fn(batch: torch.Tensor) -> torch.Tensor:
        return batch.to(device)
    return collate_fn


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, required=True , help="Path to the config file.")
    argparser.add_argument("--working_dir", type=str, required=True, help="Path to local data directory.")
    argparser.add_argument("--opt_level", type=int, default=2, help="Optimization level for equalization.")
    argparser.add_argument("--iterations", type=int, default=10, help="Number of iterations for equalization.")
    argparser.add_argument("--value_threshold", type=float, default=0.5, help="Value threshold for equalization.")
    argparser.add_argument("--including_bias", action=argparse.BooleanOptionalAction, default=False, help="Include bias in equalization.")
    argparser.add_argument("--bias_multiplier", type=float, default=0.5, help="Bias multiplier for equalization.")
    argparser.add_argument("--including_act", action=argparse.BooleanOptionalAction, default=False, help="Include activation in equalization.")
    argparser.add_argument("--act_multiplier", type=float, default=0.5, help="Activation multiplier for equalization.")
    args = argparser.parse_args()

    # Read the configuration file
    try:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
            input_model_path = os.path.join(args.working_dir, config["input_model_path"])
            dataset_path = os.path.join(args.working_dir, config["dataset_path"])
            output_path = os.path.join(args.working_dir, config["output_path"])
            os.makedirs(output_path, exist_ok=True)
            config["input_model_path"] = input_model_path
            config["dataset_path"] = dataset_path
            config["output_path"] = output_path
            _topk = config.get("topk", 1)
            topk = tuple(
                map(int, _topk if isinstance(_topk, (list, tuple)) else (_topk,))
            )
            quant_method = parse_quant_method(config["quant_config"])
        print("Configuration:", json.dumps(config, sort_keys=True, indent=4))
    except Exception as e:
        # If the configuration file is not found or invalid, print an error message and exit
        print("Error reading the config file.")
        print(e)
        exit(1)


    BATCH_SIZE = config["batch_size"]
    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
    TARGET = "esp32s3"
    NUM_OF_BITS = 8 # 8-bit quantization
    CALIB_SIZE = config.get("calib_size", 1024)
    TORCH_PATH = config["input_model_path"]
    # The equalization parameters only distinguish runs of the layerwise equalization
    if quant_method == "LayerwiseEqualization_quantization":
        RUN_NAME = f"{config['model_name']}_{args.opt_level}_{args.iterations}_{args.value_threshold}"
    else:
        RUN_NAME = config["model_name"]
    ESPDL_MODEL_PATH = os.path.join(output_path, RUN_NAME + ".espdl")
    CALIB_DIR = os.path.join(dataset_path, "train")
    print(f"Calibration Dataset Directory: {CALIB_DIR}")
    TEST_DIR = os.path.join(dataset_path, "test")

    IMAGE_HEIGHT = config.get("img_height", 224)
    IMAGE_WIDTH = config.get("img_width", 224)
    print(f"Image Size: {IMAGE_HEIGHT}x{IMAGE_WIDTH}")

    INPUT_SHAPE = [3, IMAGE_HEIGHT, IMAGE_WIDTH] # Torch format

    # -------------------------------------------
    # Prepare Calibration Dataset
    # --------------------------------------------
    if os.path.exists(CALIB_DIR):
        print(f"load calibration dataset from directory: {CALIB_DIR}")
        dataset, dataloader = load_pv_from_directory(
            directory=CALIB_DIR,
            subset=CALIB_SIZE,
            batchsize=BATCH_SIZE,
            require_label=False,
            num_of_workers=config.get("num_workers", 4),
            img_height=IMAGE_HEIGHT,
            img_width=IMAGE_WIDTH,
            seed=config.get("seed", 0),
        )
    else:
        raise ValueError("Please provide valid calibration dataset path")
    # Use every calibration image exactly once, independent of the batch size
    CALIB_STEPS = math.ceil(len(dataset) / BATCH_SIZE)
    print(f"Calibration on {len(dataset)} images in {CALIB_STEPS} steps")

    if os.path.exists(TEST_DIR):
        print(f"load testing dataset from directory: {TEST_DIR}")
        test_dataset, test_dataloader = load_pv_from_directory(
            directory=TEST_DIR,
            batchsize=BATCH_SIZE,
            require_label=True,
            num_of_workers=config.get("num_workers", 4),
            img_height=IMAGE_HEIGHT,
            img_width=IMAGE_WIDTH
        )
    else:
        raise ValueError("Provide valid testing dataset path")

    # -------------------------------------------
    # Prepare Top-k Accuracy Setting
    # --------------------------------------------

    # ImageFolder numbers the classes alphabetically per split, so a class missing in one split shifts all indices
    class_names = test_dataset.classes
    if dataset.dataset.classes != class_names:
        raise ValueError(f"train and test split have different classes:\n train: {dataset.dataset.classes}\n test: {class_names}")
    num_classes = len(class_names)
    # Define top-k accuracy and precision tag
    # raise warning if one of topk values exceed num_classes and throw it away
    topk_acc = tuple([k for k in topk if k <= num_classes])
    if len(topk_acc) < len(topk):
        print(f"Warning: Some values in topk_acc {topk} exceed the number of classes {num_classes} and will be ignored.")
    if len(topk_acc) == 0:
        raise ValueError(f"All values in topk_acc {topk} exceed the number of classes {num_classes}. Please provide valid topk_acc values.")

    # -------------------------------------------
    # Load Model
    # --------------------------------------------

    # The .pth files of the training pipeline contain the full pickled model, not only the weights
    model = torch.load(TORCH_PATH, map_location=DEVICE, weights_only=False)
    # Make sure to remove DataParallel wrapper if present
    model = model.module if isinstance(model, torch.nn.DataParallel) else model
    model.eval()
    with torch.no_grad():
        num_outputs = model(torch.zeros([1] + INPUT_SHAPE, device=DEVICE)).shape[-1]
    if num_outputs != num_classes:
        raise ValueError(f"The model has {num_outputs} outputs, but the dataset {num_classes} classes. "
                         f"Was it trained on another dataset version?")

    # -------------------------------------------
    # Set Quantization Setting
    # --------------------------------------------

    mixed_precision_layers = config.get("mixed_precision_layers")
    quant_setting = set_quant_settings(quant_method, args, TARGET, mixed_precision_layers)

    # -------------------------------------------
    # Evaluate Original Model
    # --------------------------------------------

    # Evaluate the model
    test = evaluate_torch_module_with_imagenet(
        model=model,
        batchsize=BATCH_SIZE,
        device=DEVICE,
        imagenet_validation_loader=test_dataloader,
        verbose=1,
        img_height=IMAGE_HEIGHT,
        img_width=IMAGE_WIDTH,
        print_confusion_matrix=True,
        confusion_matrix_path=os.path.join(output_path, config["model_name"] + "_before_quantization_confusion_matrix.png"),
        topk=topk_acc,
    )

    # -------------------------------------------
    # Quantize Model
    # --------------------------------------------

    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=CALIB_STEPS,
        input_shape=[1] + INPUT_SHAPE,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        collate_fn=make_collate_fn(DEVICE),
        setting=quant_setting,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=False,
        verbose=1,
    )

    # Class names for the firmware, belonging to this .espdl file
    header_path = os.path.join(output_path, RUN_NAME + "_classification_category_name.hpp")
    write_class_names_header(class_names, header_path)
    print(f"Class names for the firmware written to {header_path}")

    # The dispatching table silently ignores names that are not in the graph
    if quant_method == "MixedPrecision_quantization":
        missing = [layer for layer in mixed_precision_layers if layer not in quant_ppq_graph.operations]
        if missing:
            print(f"WARNING: mixed_precision_layers not found in the graph, they were quantized with {NUM_OF_BITS} bit: {missing}\n"
                  f"Available operations: {sorted(quant_ppq_graph.operations)}")

    # -------------------------------------------
    # Evaluate Quantized Model
    # --------------------------------------------

    quant_test = evaluate_ppq_module_with_pv(
        model=quant_ppq_graph,
        imagenet_validation_loader=test_dataloader,
        batchsize=BATCH_SIZE,
        device=DEVICE,
        verbose=1,
        print_confusion_matrix=True,
        img_height=IMAGE_HEIGHT,
        img_width=IMAGE_WIDTH,
        confusion_matrix_path=os.path.join(output_path, RUN_NAME + "_confusion_matrix.png"),
        topk=topk_acc,
    )

    results = {}
    for k in topk_acc:
        prec_tag = f"top{k}_accuracy"
        results[prec_tag + "_test"] = [sum(test[prec_tag]) / len(test[prec_tag])]
        results[prec_tag + "_test_quant"] = [sum(quant_test[prec_tag]) / len(quant_test[prec_tag])]
        print(f"{prec_tag} before quantization: {results[prec_tag+'_test'][0]:.2f}%")
        print(f"{prec_tag} after quantization: {results[prec_tag+'_test_quant'][0]:.2f}%")

    results.update({
        "model_name": config["model_name"],
        "espdl_file": os.path.basename(ESPDL_MODEL_PATH),
        "quant_config": quant_method,
        "batch_size": BATCH_SIZE,
        "calib_size": len(dataset),
        "opt_level": args.opt_level,
        "iterations": args.iterations,
        "value_threshold": args.value_threshold,
        "including_bias": bool(args.including_bias),
        "bias_multiplier": args.bias_multiplier,
        "including_act": bool(args.including_act),
        "act_multiplier": args.act_multiplier,
    })

    # concatenate the results and export to a csv file
    output_csv_path = os.path.join(output_path, config["model_name"] + "_quant-metrics.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, mode="a", header=not os.path.exists(output_csv_path), index=False)
