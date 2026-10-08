"""Build the bird species dataset from the WieDiversIstMeinGarten (Birdiary) API in one go.

Steps (all cached in --workdir, so the script can be re-run or resumed after a crash/requeue):
  metadata  fetch stations and validated movements                      -> stations.csv, movements.csv
  split     clean labels, build class map, cap per station, split, select -> class_map.csv, videos_split.csv, selected_videos.csv
  download  download the selected videos                                 -> videos/<station>_<mov>.mp4
  frames    extract frames                                               -> frames/<station>_<mov>/*.jpg
  filter    score frames with the background/unclear/bird model          -> frames.csv
  assemble  build dataset/<split>/<class>/ (background as own class)     -> dataset/, rejected/, dataset_manifest.csv
  report    statistics, leakage checks and plots                         -> reports/

Example (pilot with 20 videos per species):
  python build_dataset.py --workdir data_pilot --videos-per-species 20
Run only some steps, e.g. on a GPU node:
  python build_dataset.py --workdir data_v7 --steps filter assemble report
"""
import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from get_movements import BASE_URL, process_movements, fetch_with_retries

STEPS = ["metadata", "split", "download", "frames", "filter", "assemble", "report"]
NON_BIRD_CLASSES = {"homo_sapiens", "sciurus_vulgaris"}  # own class if supported, otherwise dropped
UNKNOWN_CLASS = "unknown_bird"
BACKGROUND_CLASS = "background"
SPECIES_SYNONYMS = {"corvus_monedula": "coloeus_monedula", "human": "homo_sapiens"}
PROB_COLS = ["p_background", "p_unclear", "p_bird"]

log = logging.getLogger("build_dataset")


def progress(iterable, **kwargs):
    # Progress bars only on interactive terminals, not in Slurm log files
    return tqdm(iterable, disable=None, mininterval=5, **kwargs)


def make_session(user_agent):
    session = requests.Session()
    if user_agent:
        session.headers.update({"User-Agent": user_agent})
    adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def read_required(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}, run the previous steps first.")
    return pd.read_csv(path)


# ---- Step: metadata ----------------------------------------------------------

def fetch_stations(session):
    response = fetch_with_retries(session, f"{BASE_URL}/station")
    if response is None:
        raise RuntimeError("Could not fetch station list")
    rows = []
    for station in response.json():
        location = station.get("location") or {}
        last_movement = station.get("lastMovement") or {}
        rows.append({
            "station_id": station["station_id"],
            "name": station.get("name"),
            "lat": location.get("lat"),
            "lng": location.get("lng"),
            "last_movement_date": last_movement.get("createdAt"),
        })
    return pd.DataFrame(rows)


def fetch_movements(session, stations, movements_per_station):
    rows = []
    for _, station in progress(stations.iterrows(), total=len(stations), desc="stations"):
        url = f"{BASE_URL}/movement/{station['station_id']}"
        if movements_per_station:
            url += f"?movements={movements_per_station}"
        response = fetch_with_retries(session, url)
        if response is None:
            log.warning(f"Station {station['station_id']} failed after retries")
            continue
        for mov in response.json() or []:
            processed = process_movements(station["station_id"], mov, validated_only=True)
            if processed:
                rows.append(processed)
        time.sleep(random.uniform(1.0, 2.5))  # be nice to the API
    return pd.DataFrame(rows)


def step_metadata(args):
    stations_csv = args.workdir / "stations.csv"
    movements_csv = args.workdir / "movements.csv"

    if args.refresh_metadata or not stations_csv.exists():
        with make_session(args.user_agent) as session:
            fetch_stations(session).to_csv(stations_csv, index=False)
    stations = pd.read_csv(stations_csv)
    log.info(f"{len(stations)} stations")

    if not movements_csv.exists() and not args.refresh_metadata and args.seed_movements and args.seed_movements.exists():
        shutil.copy2(args.seed_movements, movements_csv)
        log.info(f"Seeded movements from {args.seed_movements}")

    if args.refresh_metadata or not movements_csv.exists():
        with make_session(args.user_agent) as session:
            fetch_movements(session, stations, args.movements_per_station).to_csv(movements_csv, index=False)
    movements = pd.read_csv(movements_csv)
    log.info(f"{len(movements)} validated movements from {movements['station_id'].nunique()} stations")


# ---- Step: split -------------------------------------------------------------

def clean_labels(movements):
    """Keep single-species videos and normalize the species names to folder names."""
    mov = movements.dropna(subset=["validations", "video_link"]).drop_duplicates("mov_id").copy()
    is_multi = mov["validations"].str.contains(";")
    log.info(f"Excluding {is_multi.sum()} multi-species videos")
    mov = mov[~is_multi].copy()
    mov["species"] = mov["validations"].str.strip().str.lower().str.replace(" ", "_")
    mov["species"] = mov["species"].replace(SPECIES_SYNONYMS)
    mov["video_stem"] = mov["station_id"] + "_" + mov["mov_id"]
    log.info(f"{len(mov)} single-species videos, {mov['species'].nunique()} species")
    return mov


def build_class_map(mov, min_videos, min_stations, max_station_share):
    """Own class if enough videos/stations and no dominating station, else unknown_bird (birds) or dropped (non-birds)."""
    support = mov.groupby("species").agg(
        videos=("mov_id", "size"),
        stations=("station_id", "nunique"),
        top_station_share=("station_id", lambda s: s.value_counts(normalize=True).iloc[0]),
    )
    supported = (
        (support["videos"] >= min_videos)
        & (support["stations"] >= min_stations)
        & (support["top_station_share"] <= max_station_share)
    )
    is_bird = ~support.index.isin(NON_BIRD_CLASSES)
    support["class"] = np.where(supported, support.index, np.where(is_bird, UNKNOWN_CLASS, None))
    support["status"] = np.where(supported, "keep", np.where(is_bird, "unknown", "drop"))
    return support.sort_values("videos", ascending=False)


def cap_per_station(mov, max_videos, seed):
    """Keep at most max_videos random videos per station and species."""
    mov = mov.sample(frac=1, random_state=seed)
    return mov[mov.groupby(["station_id", "species"]).cumcount() < max_videos].copy()


def assign_station_splits(mov, ratios, seed, restarts, missing_penalty=1.0):
    """Assign every station to one split, minimizing the relative deviation of all class shares from the targets.

    Local search (single station moves and pairwise swaps) with random restarts.
    """
    names = list(ratios)
    target = np.array([ratios[s] for s in names])
    counts = mov.pivot_table(index="station_id", columns="class", values="mov_id", aggfunc="size", fill_value=0)
    C = counts.to_numpy(float)                          # stations x classes (videos)
    totals = C.sum(0)
    n_stations, n_splits = C.shape[0], len(names)
    can_fill = (C > 0).sum(0) >= n_splits               # classes with enough stations to appear in every split

    def loss(S):                                        # S: classes x splits (videos)
        share = S / totals[:, None]
        missing = ((S == 0) & can_fill[:, None]).sum()
        return (((share - target) / target) ** 2).sum() + missing_penalty * missing

    rng = np.random.default_rng(seed)
    best_assignment, best_loss = None, np.inf
    for _ in range(restarts):
        a = rng.choice(n_splits, size=n_stations, p=target)
        S = np.stack([C[a == s].sum(0) for s in range(n_splits)], axis=1)
        current = loss(S)
        improved = True
        while improved:
            improved = False
            # Move single stations to another split
            for i in rng.permutation(n_stations):
                for q in range(n_splits):
                    p = a[i]
                    if q == p:
                        continue
                    S[:, p] -= C[i]
                    S[:, q] += C[i]
                    new = loss(S)
                    if new < current - 1e-12:
                        a[i], current, improved = q, new, True
                    else:
                        S[:, p] += C[i]
                        S[:, q] -= C[i]
            # Swap two stations of different splits
            for i in rng.permutation(n_stations):
                for j in range(n_stations):
                    p, q = a[i], a[j]
                    if p == q:
                        continue
                    d = C[i] - C[j]
                    S[:, p] -= d
                    S[:, q] += d
                    new = loss(S)
                    if new < current - 1e-12:
                        a[i], a[j], current, improved = q, p, new, True
                    else:
                        S[:, p] += d
                        S[:, q] -= d
        if current < best_loss:
            best_loss, best_assignment = current, a.copy()
    station_split = dict(zip(counts.index, [names[k] for k in best_assignment]))
    return mov["station_id"].map(station_split)


def assign_video_splits(mov, ratios, seed):
    """Stratified random split of the videos per class."""
    split = pd.Series(index=mov.index, dtype=object)
    for _, group in mov.groupby("class"):
        idx = group.sample(frac=1, random_state=seed).index
        bounds = np.cumsum([0] + [round(r * len(idx)) for r in ratios.values()])
        bounds[-1] = len(idx)
        for name, lo, hi in zip(ratios, bounds[:-1], bounds[1:]):
            split.loc[idx[lo:hi]] = name
    return split


def round_robin_sample(df, k, seed):
    """Take k videos, cycling through the stations for variety."""
    df = df.sample(frac=1, random_state=seed)
    df = df.assign(_rank=df.groupby("station_id").cumcount())
    return df.sort_values("_rank", kind="stable").head(k).drop(columns="_rank")


def plot_support(support, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = support["status"].map({"keep": "#2E8B57", "unknown": "#FFA500", "drop": "#B22222"})
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(support.index, support["videos"], color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("single-species videos (log)")
    ax.set_title("Support per species (green: own class, orange: unknown_bird, red: dropped), labels: stations")
    ax.tick_params(axis="x", rotation=75)
    for x, (_, row) in enumerate(support.iterrows()):
        ax.text(x, row["videos"], str(row["stations"]), ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def step_split(args):
    reports = args.workdir / "reports"
    reports.mkdir(exist_ok=True)
    mov = clean_labels(read_required(args.workdir / "movements.csv"))

    # Class map on the full metadata, so a pilot sample gets the same classes as the full dataset
    support = build_class_map(mov, args.min_videos, args.min_stations, args.max_station_share)
    support.to_csv(args.workdir / "class_map.csv")
    plot_support(support, reports / "support_per_species.png")
    log.info("Class map:\n" + support.to_markdown(floatfmt=".2f"))
    mov["class"] = mov["species"].map(support["class"])
    mov = mov[mov["class"].notna()].copy()
    unknown = mov[mov["class"] == UNKNOWN_CLASS]
    log.info(f"{support['status'].eq('keep').sum()} species classes, {UNKNOWN_CLASS}: {len(unknown)} videos / "
             f"{unknown['station_id'].nunique()} stations from {unknown['species'].nunique()} species")

    if args.max_videos_per_station:
        before = len(mov)
        mov = cap_per_station(mov, args.max_videos_per_station, args.seed)
        log.info(f"Cap of {args.max_videos_per_station} videos per station and species: kept {len(mov)} of {before} videos")

    if args.split_by == "station":
        mov["split"] = assign_station_splits(mov, args.splits, args.seed, args.split_restarts)
    else:
        mov["split"] = assign_video_splits(mov, args.splits, args.seed)

    split_table = pd.crosstab(mov["class"], mov["split"]).reindex(columns=list(args.splits), fill_value=0)
    split_share = split_table.div(split_table.sum(axis=1), axis=0)
    split_report = split_table.join(split_share, rsuffix="_share")
    split_report["max_dev"] = (split_share - pd.Series(args.splits)).abs().max(axis=1)
    split_report.to_csv(reports / "split_videos.csv")
    log.info(f"Split of videos (target {args.splits}):\n" + split_report.to_markdown(floatfmt=".2f"))
    log.info(f"Mean max. deviation from target: {split_report['max_dev'].mean():.3f}, overall shares: "
             f"{mov['split'].value_counts(normalize=True).round(3).to_dict()}")
    missing = split_table[(split_table == 0).any(axis=1)]
    if len(missing):
        log.warning(f"Classes missing in a split:\n{missing}")
    mov.to_csv(args.workdir / "videos_split.csv", index=False)

    if args.videos_per_species:
        parts = []
        for (_, split), group in mov.groupby(["species", "split"]):
            k = max(1, round(args.videos_per_species * args.splits[split]))
            parts.append(round_robin_sample(group, k, args.seed))
        selected = pd.concat(parts)
    else:
        selected = mov
    selected.to_csv(args.workdir / "selected_videos.csv", index=False)
    log.info(f"Selected {len(selected)} videos from {selected['station_id'].nunique()} stations")


# ---- Step: download ----------------------------------------------------------

def fetch_video(session, url, path, legacy_videos):
    if path.exists():
        return "cached"
    if path.name in legacy_videos:
        shutil.copy2(legacy_videos[path.name], path)
        return "reused"
    tmp = path.with_suffix(".part")
    try:
        with session.get(url, stream=True, timeout=60) as response:
            if response.status_code != 200:
                return f"http {response.status_code}"
            with open(tmp, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
        tmp.replace(path)
        return "downloaded"
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return f"error {e}"


def step_download(args):
    selected = read_required(args.workdir / "selected_videos.csv")
    video_dir = args.workdir / "videos"
    video_dir.mkdir(exist_ok=True)
    legacy = args.legacy_video_dir
    legacy_videos = {p.name: p for p in legacy.rglob("*.mp4")} if legacy and legacy.exists() else {}

    status = {}
    with make_session(args.user_agent) as session, ThreadPoolExecutor(max_workers=args.download_workers) as ex:
        futures = {
            ex.submit(fetch_video, session, row.video_link, video_dir / f"{row.video_stem}.mp4", legacy_videos): row.video_stem
            for row in selected.itertuples()
        }
        for n, fut in enumerate(progress(as_completed(futures), total=len(futures), desc="videos"), 1):
            status[futures[fut]] = fut.result()
            if n % 500 == 0:
                log.info(f"Downloaded {n}/{len(futures)} videos")

    status = pd.Series(status, dtype=object)
    log.info("Download status:\n" + status.str.split().str[0].value_counts().to_string())
    failed = status[~status.isin(["cached", "reused", "downloaded"])]
    if len(failed):
        log.warning(f"{len(failed)} downloads failed:\n{failed.head(20).to_string()}")


# ---- Step: frames ------------------------------------------------------------

def extract_frames(video_path, out_root, args):
    """Save frames that are well exposed and differ enough from the last saved frame."""
    stem = video_path.stem
    out_dir = out_root / stem
    if out_dir.exists():
        return stem, len(list(out_dir.glob("*.jpg"))), "cached"
    tmp_dir = out_root / f"{stem}.tmp"  # renamed when done, so interrupted videos are redone
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return stem, 0, "could not open"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(int(round(fps / args.frame_rate)), 1) if args.frame_rate else 1

    frame_idx = saved = 0
    prev = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            blurred = cv2.GaussianBlur(frame, (3, 3), 0)  # cancel out sensor noise
            mean, std = blurred.mean(), blurred.std()
            keep = args.min_brightness <= mean <= args.max_brightness and std >= args.min_std
            if keep and prev is not None:
                diff = cv2.absdiff(blurred, prev)
                fraction_changed = np.mean(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) > 10)
                keep = fraction_changed >= args.frac_change and diff.mean() >= args.diff_threshold
            if keep:
                cv2.imwrite(str(tmp_dir / f"{stem}_{saved:03d}.jpg"), frame)
                saved += 1
                prev = blurred
        frame_idx += 1
    cap.release()
    tmp_dir.rename(out_dir)
    return stem, saved, "extracted"


def step_frames(args):
    selected = read_required(args.workdir / "selected_videos.csv")
    video_dir, frame_dir = args.workdir / "videos", args.workdir / "frames"
    frame_dir.mkdir(exist_ok=True)
    videos = [video_dir / f"{stem}.mp4" for stem in selected["video_stem"] if (video_dir / f"{stem}.mp4").exists()]
    if len(videos) < len(selected):
        log.warning(f"{len(selected) - len(videos)} selected videos are not downloaded")

    results = []
    with ThreadPoolExecutor(max_workers=args.frame_workers) as ex:  # cv2 releases the GIL
        futures = [ex.submit(extract_frames, p, frame_dir, args) for p in videos]
        for n, fut in enumerate(progress(as_completed(futures), total=len(futures), desc="frames"), 1):
            results.append(fut.result())
            if n % 500 == 0:
                log.info(f"Extracted frames of {n}/{len(futures)} videos")
    results = pd.DataFrame(results, columns=["video_stem", "frames", "status"])
    results.to_csv(args.workdir / "frame_extraction.csv", index=False)
    log.info("Frame extraction status:\n" + results["status"].value_counts().to_string())
    log.info(f"{results['frames'].sum()} frames, {results['frames'].eq(0).sum()} videos without frames")


# ---- Step: filter ------------------------------------------------------------

def step_filter(args):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from data_cleaner import load_model, make_transform, CLASS_TO_IDX

    class FrameDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            return self.transform(Image.open(self.paths[idx]).convert("RGB"))

    selected = read_required(args.workdir / "selected_videos.csv")
    frame_dir = args.workdir / "frames"
    frames_csv = args.workdir / "frames.csv"

    rows = []
    for stem in selected["video_stem"]:
        if (frame_dir / stem).exists():
            rows.extend({"frame": p.name, "video_stem": stem} for p in sorted((frame_dir / stem).glob("*.jpg")))
    frames = pd.DataFrame(rows, columns=["frame", "video_stem"])

    # Reuse scores of earlier runs
    cached = pd.read_csv(frames_csv)[["frame", *PROB_COLS]] if frames_csv.exists() else pd.DataFrame(columns=["frame", *PROB_COLS])
    frames = frames.merge(cached, on="frame", how="left")
    frames[PROB_COLS] = frames[PROB_COLS].astype(float)
    todo = frames[frames["p_bird"].isna()]
    log.info(f"{len(frames)} frames, {len(todo)} to score")

    if len(todo):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Scoring frames on {device}")
        model = load_model(args.filter_model, device)
        paths = [frame_dir / stem / name for name, stem in zip(todo["frame"], todo["video_stem"])]
        loader = DataLoader(FrameDataset(paths, make_transform(args.filter_img_size)), batch_size=args.filter_batch_size,
                            num_workers=args.filter_workers, pin_memory=device.type == "cuda")
        order = [CLASS_TO_IDX[c] for c in ("background", "unclear", "bird")]
        probs = []
        with torch.no_grad():
            for n, x in enumerate(progress(loader, desc="filter"), 1):
                probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
                if n % 200 == 0:
                    log.info(f"Scored {n}/{len(loader)} batches")
        frames.loc[todo.index, PROB_COLS] = torch.cat(probs).numpy()[:, order] * 100

    frames["filter_label"] = np.select(
        [frames["p_background"] >= args.filter_threshold, frames["p_unclear"] >= args.filter_threshold],
        ["background", "unclear"],
        "bird",
    )
    frames = frames.merge(selected[["video_stem", "station_id", "mov_id", "species", "class", "split"]], on="video_stem")
    frames.to_csv(frames_csv, index=False)
    log.info("Filter labels per class:\n" + pd.crosstab(frames["class"], frames["filter_label"]).to_markdown())


# ---- Step: assemble ----------------------------------------------------------

def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)  # no extra disk space
    except OSError:
        shutil.copy2(src, dst)


def step_assemble(args):
    frames = read_required(args.workdir / "frames.csv")
    frame_dir = args.workdir / "frames"
    dataset_dir, rejected_dir = args.workdir / "dataset", args.workdir / "rejected"

    # Bird frames keep their class, background frames become the background class of the same split
    frames["target"] = np.where(frames["filter_label"] == "background", BACKGROUND_CLASS, frames["class"])
    frames.loc[frames["filter_label"] == "unclear", "target"] = None

    if args.max_background_per_video:
        is_bg = frames["target"] == BACKGROUND_CLASS
        shuffled = frames[is_bg].sample(frac=1, random_state=args.seed)
        keep_bg = shuffled[shuffled.groupby("video_stem").cumcount() < args.max_background_per_video].index
        frames.loc[is_bg & ~frames.index.isin(keep_bg), "target"] = None

    for d in (dataset_dir, rejected_dir):
        shutil.rmtree(d, ignore_errors=True)

    cols = ["video_stem", "frame", "split", "class", "target", "filter_label"]
    for stem, name, split, cls, target, label in progress(frames[cols].itertuples(index=False), total=len(frames), desc="assemble"):
        src = frame_dir / stem / name
        if pd.notna(target):
            link_or_copy(src, dataset_dir / split / target / name)
        elif label == "unclear":
            link_or_copy(src, rejected_dir / "unclear" / cls / name)

    frames.to_csv(args.workdir / "dataset_manifest.csv", index=False)
    log.info(f"Wrote {frames['target'].notna().sum()} frames to {dataset_dir}")


# ---- Step: report ------------------------------------------------------------

def save_sample_grid(ds, frame_dir, path, split="train", per_class=5, seed=42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = sorted(ds["target"].unique())
    fig, axes = plt.subplots(len(classes), per_class, figsize=(2.4 * per_class, 1.8 * len(classes)), squeeze=False)
    for r, cls in enumerate(classes):
        sub = ds[(ds["target"] == cls) & (ds["split"] == split)]
        sub = sub.sample(min(per_class, len(sub)), random_state=seed)
        for c in range(per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(sub):
                row = sub.iloc[c]
                img = cv2.imread(str(frame_dir / row["video_stem"] / row["frame"]))
                ax.imshow(cv2.cvtColor(cv2.resize(img, (240, 180)), cv2.COLOR_BGR2RGB))
            if c == 0:
                ax.set_title(cls, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def step_report(args):
    frames = read_required(args.workdir / "dataset_manifest.csv")
    dataset_dir, reports = args.workdir / "dataset", args.workdir / "reports"
    reports.mkdir(exist_ok=True)
    splits = list(args.splits)

    ds = frames[frames["target"].notna()]
    images = pd.crosstab(ds["target"], ds["split"]).reindex(columns=splits, fill_value=0)
    videos = ds.groupby(["target", "split"])["video_stem"].nunique().unstack(fill_value=0).reindex(columns=splits, fill_value=0)
    stations = ds.groupby(["target", "split"])["station_id"].nunique().unstack(fill_value=0).reindex(columns=splits, fill_value=0)
    report = pd.concat({"images": images, "videos": videos, "stations": stations}, axis=1).fillna(0).astype(int)
    report.columns = [f"{m}_{s}" for m, s in report.columns]
    report.to_csv(reports / "dataset_report.csv")
    log.info("Dataset:\n" + report.to_markdown())
    log.info(f"Overall images: {ds['split'].value_counts().reindex(splits).to_dict()}")

    station_multi = (ds.groupby("station_id")["split"].nunique() > 1).sum()
    video_multi = (ds.groupby("video_stem")["split"].nunique() > 1).sum()
    missing = list(images[(images == 0).any(axis=1)].index)
    nested = [p for p in dataset_dir.glob("*/*/*") if p.is_dir()]
    log.info(f"Stations in more than one split: {station_multi} (expected 0 for --split-by station)")
    log.info(f"Videos in more than one split:   {video_multi} (expected 0)")
    log.info(f"Classes missing in a split:      {missing}")
    log.info(f"Nested folders inside classes:   {len(nested)} (expected 0)")
    log.info(f"Classes (train): {sorted(p.name for p in (dataset_dir / 'train').iterdir() if p.is_dir())}")

    save_sample_grid(ds, args.workdir / "frames", reports / "samples_train.png", seed=args.seed)
    log.info(f"Reports written to {reports}")


# ---- CLI ---------------------------------------------------------------------

def optional_int(value):
    return None if value.lower() in ("none", "0") else int(value)


def optional_path(value):
    return None if value.lower() == "none" else Path(value).expanduser().resolve()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", type=Path, required=True, help="Output directory, also holds all caches.")
    p.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS, help="Steps to run (default: all).")

    g = p.add_argument_group("metadata")
    g.add_argument("--user-agent", default="", help="Custom User-Agent for API requests.")
    g.add_argument("--refresh-metadata", action="store_true", help="Re-fetch stations/movements even if cached.")
    g.add_argument("--seed-movements", type=optional_path, default=None,
                   help="Copy this movements.csv instead of crawling the API (only if the workdir has none).")
    g.add_argument("--movements-per-station", type=int, default=1_000_000, help="API limit of movements per station.")

    g = p.add_argument_group("classes and splits")
    g.add_argument("--min-videos", type=int, default=30, help="Min. single-species videos for an own class.")
    g.add_argument("--min-stations", type=int, default=5, help="Min. stations for an own class.")
    g.add_argument("--max-station-share", type=float, default=0.8, help="Max. share of videos from the biggest station.")
    g.add_argument("--max-videos-per-station", type=optional_int, default=100,
                   help="Cap of videos per station and species ('none' = no cap).")
    g.add_argument("--split-by", choices=["station", "video"], default="station")
    g.add_argument("--splits", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"), default=[0.7, 0.2, 0.1])
    g.add_argument("--split-restarts", type=int, default=20, help="Random restarts of the station split optimizer.")
    g.add_argument("--videos-per-species", type=optional_int, default=None,
                   help="Pilot: max. videos per species ('none' = all).")
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("download")
    g.add_argument("--legacy-video-dir", type=optional_path, default=None, help="Reuse videos already downloaded there.")
    g.add_argument("--download-workers", type=int, default=4)

    g = p.add_argument_group("frames")
    g.add_argument("--frame-rate", type=float, default=5, help="Frames per second to look at.")
    g.add_argument("--diff-threshold", type=float, default=20, help="Min. mean abs. difference to the last saved frame.")
    g.add_argument("--frac-change", type=float, default=0.2, help="Min. fraction of changed pixels to the last saved frame.")
    g.add_argument("--min-brightness", type=float, default=20)
    g.add_argument("--max-brightness", type=float, default=235)
    g.add_argument("--min-std", type=float, default=50, help="Min. contrast (std of pixel values).")
    g.add_argument("--frame-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))

    g = p.add_argument_group("filter")
    g.add_argument("--filter-model", type=Path, default=SCRIPT_DIR / "acc98.98_e22_bird_filter_0729_1145.pth")
    g.add_argument("--filter-img-size", type=int, default=960)
    g.add_argument("--filter-threshold", type=float, default=90.0, help="Percent probability for background/unclear.")
    g.add_argument("--filter-batch-size", type=int, default=8)
    g.add_argument("--filter-workers", type=int, default=0, help="DataLoader workers for the filter.")
    g.add_argument("--max-background-per-video", type=optional_int, default=None,
                   help="Cap of background frames per video ('none' = keep all).")

    args = p.parse_args(argv)
    if abs(sum(args.splits) - 1) > 1e-6:
        p.error("--splits must sum to 1")
    args.splits = dict(zip(["train", "val", "test"], args.splits))
    args.workdir = args.workdir.expanduser().resolve()
    args.filter_model = args.filter_model.expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    args.workdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(args.workdir / "build_dataset.log")],
    )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.workdir / "config.json").write_text(json.dumps(config, indent=2))
    log.info(f"Config: {config}")

    steps = {"metadata": step_metadata, "split": step_split, "download": step_download, "frames": step_frames,
             "filter": step_filter, "assemble": step_assemble, "report": step_report}
    for name in STEPS:  # always in pipeline order
        if name in args.steps:
            log.info(f"===== Step: {name}")
            start = time.time()
            steps[name](args)
            log.info(f"===== Step {name} done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
