import argparse
import os
import random
from pathlib import Path
import shutil
from datetime import datetime
from typing import List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models


def build_binary_filelist(path: Path) -> List[Tuple[Path, int]]:
    """Collect all image paths and assign labels: 0 for bird images under 'classes/', 1 for images under 'no_bird/'. 
    Returns a list of (path, label) tuples for training/validation."""
    no_bird_dir = path / "no_bird"
    classes_dir = path / "classes"

    if not no_bird_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {no_bird_dir}")
    if not classes_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {classes_dir}")

    files = []
    for p in list(classes_dir.rglob("*.jpg")):
        files.append((p, 0))  # bird
    for p in list(no_bird_dir.rglob("*.jpg")):
        files.append((p, 1))  # no_bird
    if not files:
        raise RuntimeError(f"No images found under {path}")
    return files


class BinaryImageDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(y, dtype=torch.long)


def stratified_split(samples: List[Tuple[Path, int]], seed):
    """Split samples into train and validation sets while preserving class ratios."""
    by_class = {0: [], 1: []}
    for s in samples:
        by_class[s[1]].append(s)
    rng = random.Random(seed)
    val_ratio = 0.2
    train, val = [], []
    for c, lst in by_class.items():
        rng.shuffle(lst)
        n_val = max(1, int(round(len(lst) * val_ratio))) if len(lst) > 0 else 0
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def make_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def train_model(args):
    """Train a ResNet-18 binary classifier (bird vs no_bird) on the dataset with a train/val split."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    path = Path(args.workdir).expanduser().resolve()
    samples = build_binary_filelist(path)
    train_s, val_s = stratified_split(samples, args.seed)

    # Wrap samples in Dataset objects
    transform = make_transform(args.img_size)
    train_ds = BinaryImageDataset(train_s, transform)
    val_ds = BinaryImageDataset(val_s, transform)

    # Create DataLoaders for batching, shuffling, prefetching
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    print(f"[INFO] Classes: ['bird', 'no_bird']")
    print(f"[INFO] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace final FC layer for binary classification (2 outputs, bird / no_bird)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # For checkpoints
    timestamp = datetime.now().strftime("%m%d_%H%M")
    best_acc = -1.0

    epochs = args.epochs
    for epoch in range(epochs):
        print(f"\n[INFO] Epoch {epoch+1}/{epochs}")
        # train
        model.train()
        running_loss = 0.0
        for bidx, (x, y) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True) # Reset gradients
            out = model(x) # Forward pass
            loss = criterion(out, y)
            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            running_loss += loss.item() # Track batch loss

            # Periodic logging
            if bidx % max(1, args.log_every) == 0 or bidx == len(train_loader):
                print(f"  [TRAIN] Batch {bidx}/{len(train_loader)} Loss: {loss.item():.4f}")

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"  [TRAIN] Average Loss: {avg_loss:.4f}")

        # val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad(): # Disables gradient tracking
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                out = model(x)
                pred = out.argmax(1) # Get predicted class index
                total += y.size(0)
                correct += (pred == y).sum().item()
        acc = 100.0 * correct / max(1, total)
        print(f"  [VAL] Accuracy: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            out_path = path / f"acc{acc:.2f}_e{epoch+1}_bird_filter_{timestamp}.pth"
            torch.save(model.state_dict(), out_path)
            print(f"  [SAVED] Model saved as {out_path.relative_to(path)}")


def load_model(model_path: Path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def run_filter(args):
    """Run the trained model on a frames folder and move predicted 'no_bird' images into a subfolder. 
    The moved files can then be manually verified and/or deleted."""
    frames_path = Path(args.frames_dir).expanduser().resolve()
    if not frames_path.is_dir():
        raise FileNotFoundError(f"--frames-dir not found: {frames_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.model_path).expanduser().resolve(), device)
    transform = make_transform(args.img_size)

    idx_to_class = {0: "bird", 1: "no_bird"}

    # iterate species dirs
    for species_dir in sorted([d for d in frames_path.iterdir() if d.is_dir()]):
        no_bird_dir = species_dir / "no_bird"
        moved_any = False

        imgs = [p for p in species_dir.glob("*.jpg")]
        for img_path in imgs:
            try:
                img = Image.open(img_path).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(x)
                    probs = torch.softmax(out, dim=1).squeeze(0)
                    p_no_bird = float(probs[1].item()) * 100.0

                if p_no_bird >= args.threshold:
                    if not moved_any:
                        no_bird_dir.mkdir(parents=True, exist_ok=True)
                        moved_any = True
                    shutil.move(str(img_path), str(no_bird_dir / img_path.name))
                    print(f"[MOVED] {img_path.relative_to(frames_path)} "
                          f"-> {(no_bird_dir).relative_to(frames_path)} "
                          f"(P(no_bird)={p_no_bird:.1f}%)")
            except Exception as e:
                print(f"[ERROR] Failed on {img_path}: {e}")



def main():
    parser = argparse.ArgumentParser(description="Train or run a binary bird/no_bird classifier.")
    sub = parser.add_subparsers(dest="mode", required=True)

    # train
    pt = sub.add_parser("train", help="Train bird/no_bird model from path(no_bird/, classes/)")
    pt.add_argument("--workdir", type=str, required=True, default="./",
                    help="Path containing 'no_bird/' and 'classes/' directories.")
    pt.add_argument("--img-size", type=int, default=224)
    pt.add_argument("--batch-size", type=int, default=32)
    pt.add_argument("--epochs", type=int, default=10)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    pt.add_argument("--log-every", type=int, default=50)

    # run
    pr = sub.add_parser("run", help="Run model over frames_dir and move 'no_bird' frames.")
    pr.add_argument("--model-path", type=str, required=True)
    pr.add_argument("--frames-dir", type=str, required=True,
                    help="Path with species subfolders that contain *.jpg frames.")
    pr.add_argument("--img-size", type=int, default=224)
    pr.add_argument("--threshold", type=float, default=90.0, help="Only move images if P(no_bird) ≥ this value. Default: 90.")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(args)
    elif args.mode == "run":
        run_filter(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    

if __name__ == "__main__":
    main()
