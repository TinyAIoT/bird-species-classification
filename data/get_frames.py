import os
import argparse
from pathlib import Path
import random
import cv2
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed


def collect_videos(video_base_dir):
    """Collect all videos grouped by species."""
    videos = []
    # Traverse species directories
    for species_dir in video_base_dir.iterdir():
        if species_dir.is_dir():
            # Iterate over mp4 files in the species directory
            for video_path in species_dir.glob("*.mp4"):
                videos.append((species_dir.name, video_path))
    return videos


def split_videos(splits, videos):
    """Shuffle and split videos into train/test/val."""
    # Splits done on video basisinstead of frame to avoid mixing neighboring frames and causing knowledge leakage
    random.seed(42)
    random.shuffle(videos)
    n_total = len(videos)
    n_train = int(n_total * splits["train"])
    n_val = int(n_total * splits["val"])
    return {
        "train": videos[:n_train],
        "val": videos[n_train:n_train+n_val],
        "test": videos[n_train+n_val:]
    }


def extract_frames(split_map, workdir, frame_rate, workers):
    """Run frame extraction in parallel."""
    futures = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for split, vid_list in split_map.items():
            for species, video_path in vid_list:
                futures.append(ex.submit(process_video, str(video_path), split, species, workdir, frame_rate))

        for fut in as_completed(futures):
            video_path, saved_count = fut.result()
            if saved_count > 0:
                print(f"[OK] Extracted {saved_count} frames from {Path(video_path).relative_to(workdir)}.")
            else:
                print(f"[ERROR] Could not open {video_path}.")


def process_video(video_path, split, species, workdir, frame_rate):
    """Extract frames from one video into dataset/<split>/<species> and full_dataset/<species>."""
    try:
        cv2.setNumThreads(1) # keep OpenCV single-threaded inside each process
    except Exception:
        pass

    dataset_dir = workdir / "dataset" / split / species
    full_dataset_dir = workdir / "full_dataset" / species
    dataset_dir.mkdir(parents=True, exist_ok=True)
    full_dataset_dir.mkdir(parents=True, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (str(video_path), 0)

    # Determine frame capture interval
    if frame_rate and frame_rate > 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30   # Get frame rate or default to 30 fps
        frame_interval = max(int(round(fps / frame_rate)), 1) # e.g. video frame rate 30, get 5 frames per second => save every 6th frame
    else:
        frame_interval = 1  # capture all frames

    frame_idx = 0
    saved_count = 0
    while True:
        # Read next frame from video
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            filename = f"{Path(video_path).stem}_{saved_count:03d}.jpg"
            # Save twice: once into a split and once to full dataset
            cv2.imwrite(str(dataset_dir / filename), frame)
            cv2.imwrite(str(full_dataset_dir / filename), frame)
            saved_count += 1
        frame_idx += 1

    cap.release()
    return (str(video_path), saved_count)


def main():
    parser = argparse.ArgumentParser(description="Split videos into frames.")
    parser.add_argument("--workdir", type=str, default="./", help="Path to local data directory.")
    parser.add_argument("--frame-rate", type=float, default=5, help="Number of frames-per-second to capture.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of parallel worker processes.")
    parser.add_argument("--train", type=float, default=0.7, help="Proportion for training set.")
    parser.add_argument("--val", type=float, default=0.2, help="Proportion for validation set.")
    parser.add_argument("--test", type=float, default=0.1, help="Proportion for test set.")
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # Input and output base directories
    video_base_dir = workdir / "videos"

    splits = {"train": args.train, "val": args.val, "test": args.test}
    if abs(sum(splits.values()) - 1.0) > 1e-6:
        raise ValueError("Train/val/test proportions must sum to 1.")
    
    videos = collect_videos(video_base_dir)
    if not videos:
        print("[ERROR] No videos found.")

    split_map = split_videos(splits, videos)
    print(f"[INFO] Total videos: {len(videos)}\t(train={len(split_map['train'])}, "
          f"val={len(split_map['val'])}, test={len(split_map['test'])})")
    
    extract_frames(split_map, workdir, args.frame_rate, args.workers)


if __name__ == "__main__":
    main()
