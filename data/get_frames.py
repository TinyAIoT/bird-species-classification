import os
import argparse
from pathlib import Path
import random
import cv2
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


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


def extract_frames(split_map, workdir, frame_rate, diff_threshold, frac_change, workers):
    """Run frame extraction in parallel."""
    futures = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for split, vid_list in split_map.items():
            for species, video_path in vid_list:
                futures.append(ex.submit(process_video, str(video_path), split, species, workdir, frame_rate, diff_threshold, frac_change))

        for fut in as_completed(futures):
            video_path, saved_count, skipped= fut.result()
            if saved_count > 0:
                print(f"[OK] Extracted {saved_count} frames from {Path(video_path).relative_to(workdir / 'videos')}. Skipped {skipped}")
                continue
            else:
                print(f"[ERROR] Could not open {video_path}.")


def process_video(video_path, split, species, workdir, frame_rate, diff_threshold, frac_change):
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
    skip_count = 0
    frac_skip_count = 0
    mean_diff_skip_count = 0
    prev_img_avg = None

    while True:
        # Read next frame from video
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            #img_avg = frame
            img_avg = cv2.GaussianBlur(frame, (3, 3), 0)  # To cancel out changes due to noise in images
            if img_avg.mean() < 20 or img_avg.mean() > 235 or img_avg.std() < 50:
                # Skip almost white/black frames and very low contrast images
                if img_avg.std() < 20:
                    # For debugging
                    filename = f"{Path(video_path).stem}_{saved_count:03d}.jpg"
                    cv2.imwrite(str(filename), frame)
                
                skip_count += 1
                frame_idx += 1
                continue

            if prev_img_avg is not None:
                # Mean absolute pixel difference
                diff = cv2.absdiff(img_avg, prev_img_avg)
                mean_diff = diff.mean()

                # Count fraction of pixels that differ by more than 10 intensity levels
                grey_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                changed = np.sum(grey_diff > 10)
                fraction_changed = changed / grey_diff.size

                if fraction_changed < frac_change:
                    # Skip frame — not enough pixel level changes
                    frame_idx += 1
                    frac_skip_count += 1
                    continue

                if mean_diff < diff_threshold:
                    # Skip frame — too similar to previous
                    frame_idx += 1
                    mean_diff_skip_count += 1
                    continue

            # Save current frame (two copies)
            filename = f"{Path(video_path).stem}_{saved_count:03d}.jpg"
            cv2.imwrite(str(dataset_dir / filename), frame)
            cv2.imwrite(str(full_dataset_dir / filename), frame)
            saved_count += 1
            prev_img_avg = img_avg  

        frame_idx += 1

    cap.release()   
    return (str(video_path), saved_count, (frac_skip_count, mean_diff_skip_count, skip_count))


def main():
    parser = argparse.ArgumentParser(description="Split videos into frames.")
    parser.add_argument("--workdir", type=str, default="./", help="Path to local data directory.")
    parser.add_argument("--frame-rate", type=float, default=5, help="Number of frames-per-second to capture.")
    parser.add_argument("--diff-threshold", type=float, default=20, help="Threshold for image similarity (change in avg channel intensity).")
    parser.add_argument("--frac-change", type=float, default=0.2, help="Fraction of changed pixels between images.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of parallel worker processes.")
    parser.add_argument("--split", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"), default=[0.7, 0.2, 0.1], help="Train, val, and test split proportions (must sum to 1.0)")
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # Input and output base directories
    video_base_dir = workdir / "videos"

    train, val, test = args.split
    if abs(sum(args.split) - 1.0) > 1e-6:
        raise ValueError("Train/val/test proportions must sum to 1.")
    splits = {"train": train, "val": val, "test": test}
    
    videos = collect_videos(video_base_dir)
    if not videos:
        print("[ERROR] No videos found.")

    split_map = split_videos(splits, videos)
    print(f"[INFO] Total videos: {len(videos)}\t(train={len(split_map['train'])}, "
          f"val={len(split_map['val'])}, test={len(split_map['test'])})")
    
    extract_frames(split_map, workdir, args.frame_rate, args.diff_threshold, args.frac_change, args.workers)


if __name__ == "__main__":
    main()
