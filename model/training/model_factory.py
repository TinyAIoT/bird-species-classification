import torch
import torch.nn as nn
import torchvision
import os
import glob


def load_weights(model, checkpoint_path, strict=False):
    """Load the weights of a checkpoint into model (in place).

    Accepts state dicts (.pt) and full pickled models (.pth), with or without DataParallel wrapper on either side.
    With strict=False, parameters whose shape differs from the model (e.g. the classification head of a model trained
    with a different number of classes) are skipped and keep their current values.
    Args:
        model (torch.nn.Module): Model to load the weights into
        checkpoint_path (str): Path to the checkpoint
        strict (bool): Require that all parameters match exactly
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        checkpoint = checkpoint.state_dict()
    state_dict = {k.removeprefix("module."): v for k, v in checkpoint.items()}
    target = model.module if isinstance(model, nn.DataParallel) else model

    if strict:
        target.load_state_dict(state_dict, strict=True)
        return

    own_state = target.state_dict()
    skipped = [k for k, v in state_dict.items() if k in own_state and own_state[k].shape != v.shape]
    state_dict = {k: v for k, v in state_dict.items() if k not in skipped}
    result = target.load_state_dict(state_dict, strict=False)
    loaded = sum(k in own_state for k in state_dict)
    if loaded == 0:
        raise ValueError(f"No parameters of {checkpoint_path} match the model. Is it the same architecture?")
    print(f"Loaded {loaded}/{len(own_state)} parameters from {checkpoint_path}")
    if skipped:
        print(f"  Skipped (different shape, e.g. other number of classes): {skipped}")
    if result.missing_keys:
        print(f"  Not in checkpoint: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Not in model: {result.unexpected_keys}")


class ModelFactory:
    """Factory class for creating and configuring different model architectures"""

    def __init__(self, config):
        """Initialize the ModelFactory with configuration parameters
        Args:
            config (dict): Dictionary containing configuration parameters
        """
        self.model_type = config["model_type"]
        self.requires_grad = config.get("requires_grad", False)
        self.weights_path = config.get("weights_path", None)
        # Architecture of the checkpoint for model_type "from_checkpoint" (inferred from the checkpoint's YAML if not given)
        self.base_model_type = config.get("base_model_type", None)

    def _builders(self):
        return {
            "shufflenet": self._create_shufflenet,
            "mobilenetv2": self._create_mobilenetv2,
            "mobilenetv3": self._create_mobilenetv3,
            "efficientnet": self._create_efficientnet_v2_s,
            "efficientnet_b0": self._create_efficientnet_b0,
            "densenet": self._create_densenet,
            "squeezenet": self._create_squeezenet,
            "mnasnet": self._create_mnasnet,
        }

    def create_model(self, num_classes, device):
        """Create and configure the model based on the specified type
        Args:
            num_classes (int): Number of output classes for the model
            device (torch.device): Device to move the model to (CPU or GPU)
        Returns:
            model (torch.nn.Module): Configured PyTorch model
        """

        if self.model_type == "from_checkpoint":
            model = self._create_from_checkpoint(num_classes, self.weights_path)
        elif self.model_type in self._builders():
            model = self._builders()[self.model_type](num_classes, self.weights_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types: {', '.join(self._builders())}, from_checkpoint")

        print(model)
        # Set initial gradient requirements
        for param in model.parameters():
            param.requires_grad = self.requires_grad

        # Move to device and handle multi-GPU
        print(f"Using {device} for training.")

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training.")
            model = nn.DataParallel(model)
        elif torch.backends.mps.is_available() and torch.mps.device_count() > 1:
            print(f"Using {torch.mps.device_count()} MPS for training.")
            model = nn.DataParallel(model)

        model = model.to(device)
        return model

    def _create_shufflenet(self, num_classes, weights_path):
        """Create ShuffleNet V2 1.5x model
        Args:
            num_classes (int): Number of output classes for the model
        Returns:
            model (torch.nn.Module): ShuffleNet V2 1.5x model with modified classifier
        """
        model = torchvision.models.shufflenet_v2_x1_5(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if weights_path is not None:
            load_weights(model, weights_path)
        return model

    def _create_mobilenetv2(self, num_classes, weights_path):
        """Create MobileNet V2 model
        Hint: Relaces ReLU6 with ReLU for better compatibility.
        Args:
            num_classes (int): Number of output classes for the model
        Returns:
            model (torch.nn.Module): MobileNet V2 model with modified classifier
        """
        model = torchvision.models.mobilenet_v2(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        # Convert ReLU6 to ReLU for better compatibility
        model = self._convert_relu6_to_relu(model)
        if weights_path is not None:
            load_weights(model, weights_path)
        return model

    def _create_mobilenetv3(self, num_classes, weights_path):
        """Create MobileNet V3 S model
        Hint: Relaces ReLU6 with ReLU for better compatibility.
        Args:
            num_classes (int): Number of output classes for the model
        Returns:
            model (torch.nn.Module): MobileNet V3 model with modified classifier
        """
        model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

        # Convert ReLU6 to ReLU for better compatibility
        model = self._convert_relu6_to_relu(model)
        if weights_path is not None:
            load_weights(model, weights_path)
        return model

    def _create_efficientnet_v2_s(self, num_classes, weights_path):
        """Create EfficientNet V2 Small model
        Args:
            num_classes (int): Number of output classes for the model
        Returns:
            model (torch.nn.Module): EfficientNet V2 Small model with modified classifier
        """
        model = torchvision.models.efficientnet_v2_s(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        if weights_path is not None:
            load_weights(model, weights_path)
        return model

    def _create_efficientnet_b0(self, num_classes, weights_path):
        """Create EfficientNet B0 model
        Args:
            num_classes (int): Number of output classes for the model
        Returns:
            model (torch.nn.Module): EfficientNet B0 model with modified classifier
        """
        model = torchvision.models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        if weights_path is not None:
            load_weights(model, weights_path)
        return model

    def _create_densenet(self, num_classes, weights_path):
        """Create DenseNet 121 model
        Args:
            num_classes (int): Number of output classes for the model
        Returns:
            model (torch.nn.Module): DenseNet 121 model with modified classifier
        """
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        if weights_path is not None:
            load_weights(model, weights_path)
        return model

    def _create_squeezenet(self, num_classes, weights_path):
        """Create SqueezeNet 1.1 model
        Args:
            num_classes (int): Number of output classes for the model
        Returns:
            model (torch.nn.Module): SqueezeNet 1.1 model with modified classifier
        """

        model = torchvision.models.squeezenet1_1(weights="DEFAULT")
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        if weights_path is not None:
            load_weights(model, weights_path)
        return model

    def _create_mnasnet(self, num_classes, weights_path):
        """Create MNASNet 0.5 model
        Args:
            num_classes (int): Number of output classes for the model
        Returns:
            model (torch.nn.Module): MNASNet 0.5 model with modified classifier
        """
        model = torchvision.models.mnasnet0_5(weights="DEFAULT")
        model.classifier = nn.Linear(model.classifier[1].in_features, num_classes)
        if weights_path is not None:
            load_weights(model, weights_path)
        return model

    def _create_from_checkpoint(self, num_classes, weights_path):
        """Build the architecture of an earlier training run and initialize it with its weights (except a
        classification head with a different number of classes), e.g. to continue training on a new dataset version.

        The architecture is taken from base_model_type in the config or, if not given, from the model_type in the
        training config saved next to the checkpoint (<output_path>/<model_name>/<model_name>.yaml).
        Args:
            num_classes (int): Number of output classes for the model
            weights_path (str): Checkpoint of the earlier run (.pt state dict or .pth full model)
        Returns:
            model (torch.nn.Module): Model initialized from the checkpoint
        """
        if weights_path is None:
            raise ValueError("model_type 'from_checkpoint' needs weights_path in the config.")
        base_type = self.base_model_type or self._infer_model_type(weights_path)
        base_type = {"mobilenet": "mobilenetv2"}.get(base_type, base_type)  # name used in older configs
        if base_type not in self._builders():
            raise ValueError(f"Unsupported base model type '{base_type}' for {weights_path}. "
                             f"Set base_model_type in the config to one of: {', '.join(self._builders())}")
        print(f"Creating {base_type} from checkpoint {weights_path}")
        return self._builders()[base_type](num_classes, weights_path)

    @staticmethod
    def _infer_model_type(weights_path):
        """Read model_type from the training config saved next to a checkpoint"""
        import yaml

        run_dir = os.path.dirname(os.path.abspath(weights_path))
        candidates = [os.path.join(run_dir, os.path.basename(run_dir) + ".yaml"),
                      os.path.splitext(weights_path)[0] + ".yaml"]
        yaml_files = glob.glob(os.path.join(run_dir, "*.yaml"))
        if len(yaml_files) == 1:  # e.g. a renamed run directory
            candidates.append(yaml_files[0])
        for yaml_path in candidates:
            if os.path.exists(yaml_path):
                with open(yaml_path, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                if cfg.get("model_type") == "from_checkpoint":
                    return cfg.get("base_model_type")
                if "model_type" in cfg:
                    return cfg["model_type"]
        raise ValueError(f"Cannot infer the architecture of {weights_path}: no training config with model_type found "
                         f"({', '.join(candidates)}). Set base_model_type in the config.")

    def load_checkpoint(self, model, checkpoint_path):
        """Load the weights of a checkpoint of the same model into model (in place)
        Args:
            model: The model to load the state into
            checkpoint_path (str): Path to the checkpoint file (.pt state dict or .pth full model).
        Returns:
            bool: True if model was loaded successfully, False otherwise"""
        if os.path.exists(checkpoint_path):
            load_weights(model, checkpoint_path, strict=True)
            print(f"Model loaded from {checkpoint_path}")
            return True
        else:
            print(f"Checkpoint not found at {checkpoint_path}")
            return False

    def _convert_relu6_to_relu(self, model):
        """Convert ReLU6 layers to ReLU layers in the model
        Args:
            model (torch.nn.Module): The model to convert
        Returns:
            model (torch.nn.Module): Model with ReLU6 layers replaced by ReLU
        """
        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU6):
                inplace = child.inplace
                setattr(model, child_name, nn.ReLU(inplace=inplace))
            else:
                self._convert_relu6_to_relu(child)
        return model

    def get_model_info(self):
        """Get information about the model type
        Returns:
            str: Description of the model type
        """
        model_info = {
            "shufflenet": "ShuffleNet V2 1.5x - Lightweight CNN for mobile devices",
            "mobilenetv2": "MobileNet V2 - Efficient CNN for mobile and embedded devices",
            "mobilenetv3": "MobileNet V3 Small - Efficient CNN for mobile and embedded devices",
            "efficientnet": "EfficientNet V2 Small - Balanced accuracy and efficiency",
            "efficientnet_b0": "EfficientNet B0 - Base EfficientNet model",
            "densenet": "DenseNet 121 - Dense connections for better feature reuse",
            "squeezenet": "SqueezeNet 1.1 - Very lightweight CNN",
            "mnasnet": "MNASNet 0.5 - Neural architecture search optimized model",
            "from_checkpoint": f"{self.base_model_type or 'Architecture'} initialized from {self.weights_path}",
        }
        return model_info.get(self.model_type, "Unknown model type")
