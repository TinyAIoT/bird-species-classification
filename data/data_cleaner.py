import argparse
import os
import random
from pathlib import Path
import shutil
from datetime import datetime
from typing import List, Tuple, Dict

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms, models


# ---- Data --------------------------------------------------------------------

# Class index mapping (stable & explicit)
IDX_TO_CLASS = {0: "background", 1: "unclear", 2: "bird"}
CLASS_TO_IDX = {v: k for k, v in IDX_TO_CLASS.items()}

def build_class_filelist(path: Path) -> List[Tuple[Path, int]]:
    """
    Collect all image paths and assign labels:
      0 = background (path/background/*.jpg or subfolders)
      1 = unclear    (path/unclear/*.jpg or subfolders)
      2 = bird       (path/classes/*.jpg or subfolders)

    Returns a list of (path, label) tuples for training/validation.
    """
    background_dir = path / "background"
    unclear_dir = path / "unclear"
    bird_dir = path / "classes"

    for d in (background_dir, unclear_dir, bird_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Missing directory: {d}")

    files: List[Tuple[Path, int]] = []
    for p in background_dir.rglob("*.jpg"):
        files.append((p, CLASS_TO_IDX["background"]))
    for p in unclear_dir.rglob("*.jpg"):
        files.append((p, CLASS_TO_IDX["unclear"]))
    for p in bird_dir.rglob("*.jpg"):
        files.append((p, CLASS_TO_IDX["bird"]))

    if not files:
        raise RuntimeError(f"No images found under {path}")
    return files


class ImageDataset(Dataset):
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
    
def make_weighted_sampler(train_samples: List[Tuple[Path, int]]) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that balances classes by sampling inversely
    proportional to class frequency. Keeps epoch length at len(train_samples).
    """
    # count per class
    counts: Dict[int, int] = {}
    for _, y in train_samples:
        counts[y] = counts.get(y, 0) + 1

    # inverse frequency weight per class (protect against div-by-zero)
    class_weight = {c: 1.0 / max(1, n) for c, n in counts.items()}

    # per-sample weights
    sample_weights = [class_weight[y] for _, y in train_samples]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    # with replacement=True to allow oversampling minority classes
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_samples),
        replacement=True,
    )
    return sampler



def stratified_split(samples: List[Tuple[Path, int]], seed: int, val_ratio: float = 0.2):
    """
    Split samples into train/val preserving class ratios for arbitrary class indices.
    """
    by_class: Dict[int, List[Tuple[Path, int]]] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)

    rng = random.Random(seed)
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


# ---- Training ----------------------------------------------------------------

def _count_by_class(samples: List[Tuple[Path, int]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for _, y in samples:
        counts[y] = counts.get(y, 0) + 1
    return counts


def train_model(args):
    """
    Train a ResNet-18 3-class classifier: background / unclear / bird.
    """
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    path = Path(args.workdir).expanduser().resolve()
    samples = build_class_filelist(path)
    train_s, val_s = stratified_split(samples, args.seed, val_ratio=0.2)

    transform = make_transform(args.img_size)
    train_ds = ImageDataset(train_s, transform)
    val_ds = ImageDataset(val_s, transform)

    if args.no_sampler:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers, pin_memory=True
        )
        print("[INFO] Using plain shuffled DataLoader (no sampler).")
    else:
        sampler = make_weighted_sampler(train_s)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            sampler=sampler, shuffle=False,  # don't set shuffle with sampler
            num_workers=args.workers, pin_memory=True
        )
        print("[INFO] Using WeightedRandomSampler for class balance.")

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )


    # Log dataset stats
    print(f"[INFO] Classes: { [IDX_TO_CLASS[i] for i in sorted(IDX_TO_CLASS)] }")
    tr_counts = _count_by_class(train_s)
    va_counts = _count_by_class(val_s)
    for i in sorted(IDX_TO_CLASS):
        print(f"  [INFO] Train {IDX_TO_CLASS[i]}: {tr_counts.get(i,0)} | "
              f"Val {IDX_TO_CLASS[i]}: {va_counts.get(i,0)}")
    print(f"[INFO] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

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
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if bidx % max(1, args.log_every) == 0 or bidx == len(train_loader):
                print(f"  [TRAIN] Batch {bidx}/{len(train_loader)} Loss: {loss.item():.4f}")

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"  [TRAIN] Average Loss: {avg_loss:.4f}")

        # val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                out = model(x)
                pred = out.argmax(1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        acc = 100.0 * correct / max(1, total)
        print(f"  [VAL] Accuracy: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            out_path = path / f"acc{acc:.2f}_e{epoch+1}_bird_filter_{timestamp}.pth"
            torch.save(model.state_dict(), out_path)
            print(f"  [SAVED] Model saved as {out_path.relative_to(path)}")


# ---- Inference / Filtering ----------------------------------------------------

def load_model(model_path: Path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def run_filter(args):
    """
    Run the trained 3-class model over --frames-dir.
    For each species subfolder, move images predicted as background/unclear with
    probability ≥ --threshold to subfolders 'background' or 'unclear'.
    """
    frames_path = Path(args.frames_dir).expanduser().resolve()
    if not frames_path.is_dir():
        raise FileNotFoundError(f"--frames-dir not found: {frames_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.model_path).expanduser().resolve(), device)
    transform = make_transform(args.img_size)

    # iterate species dirs
    species_dirs = [d for d in frames_path.iterdir() if d.is_dir()]
    for species_dir in sorted(species_dirs):
        moved_bg = moved_un = False
        imgs = [p for p in species_dir.glob("*.jpg")]

        for img_path in imgs:
            try:
                img = Image.open(img_path).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(x)
                    probs = torch.softmax(out, dim=1).squeeze(0).cpu()

                p_bg = float(probs[CLASS_TO_IDX["background"]]) * 100.0
                p_un = float(probs[CLASS_TO_IDX["unclear"]]) * 100.0

                if p_bg >= args.threshold:
                    dst = species_dir / "background"
                    if not moved_bg:
                        dst.mkdir(parents=True, exist_ok=True)
                        moved_bg = True
                    shutil.move(str(img_path), str(dst / img_path.name))
                    rel = dst.relative_to(frames_path)
                    print(f"[MOVED] {img_path.relative_to(frames_path)} -> {rel} (P(background)={p_bg:.1f}%)")
                elif p_un >= args.threshold:
                    dst = species_dir / "unclear"
                    if not moved_un:
                        dst.mkdir(parents=True, exist_ok=True)
                        moved_un = True
                    shutil.move(str(img_path), str(dst / img_path.name))
                    rel = dst.relative_to(frames_path)
                    print(f"[MOVED] {img_path.relative_to(frames_path)} -> {rel} (P(unclear)={p_un:.1f}%)")

            except Exception as e:
                print(f"[ERROR] Failed on {img_path}: {e}")


# ---- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train or run a 3-class (background/unclear/bird) image classifier.")
    sub = parser.add_subparsers(dest="mode", required=True)

    # train
    pt = sub.add_parser("train", help="Train model from path(background/, unclear/, bird/)")
    pt.add_argument("--workdir", type=str, required=True, help="Path containing 'background/', 'unclear/', and 'bird/' directories.")# in the 'train' subparser
    pt.add_argument("--no-sampler", action="store_true", help="Disable class-balanced WeightedRandomSampler and use plain shuffling.")
    pt.add_argument("--img-size", type=int, default=960)
    pt.add_argument("--batch-size", type=int, default=32)
    pt.add_argument("--epochs", type=int, default=10)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    pt.add_argument("--log-every", type=int, default=50)

    # run
    pr = sub.add_parser("run", help="Run model over frames_dir and move 'background'/'unclear' frames.")
    pr.add_argument("--model-path", type=str, required=True)
    pr.add_argument("--frames-dir", type=str, required=True,
                    help="Path with species subfolders that contain *.jpg frames.")
    pr.add_argument("--img-size", type=int, default=960)
    pr.add_argument("--threshold", type=float, default=90.0, help="Move to background/unclear if P(class) ≥ threshold (default: 90).")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(args)
    elif args.mode == "run":
        run_filter(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    

if __name__ == "__main__":
    main()
