import argparse, os, re, shutil, sys
from pathlib import Path
import pandas as pd


def looks_like_path(series: pd.Series) -> bool:
    if series.dtype != object:
        return False
    s = series.astype(str)
    return s.str.contains(r"[\\/]").any()

def parse_issue_columns(cols):
    """
    Returns dict:
      issue_name -> {"score": "<colname or None>", "flag": "<boolean colname or None>"}
    Example: low_information -> {"score": "low_information_score", "flag": "is_low_information_issue"}
    """
    issues = {}
    score_pat = re.compile(r"^(?P<name>.+)_score$")
    flag_pat  = re.compile(r"^is_(?P<name>.+)_issue$")
    for c in cols:
        m = score_pat.match(c)
        if m:
            name = m.group("name")
            issues.setdefault(name, {})["score"] = c
        m = flag_pat.match(c)
        if m:
            name = m.group("name")
            issues.setdefault(name, {})["flag"] = c
    # Keep only those that have at least a flag or a score
    return {k:v for k,v in issues.items() if ("score" in v) or ("flag" in v)}

def parse_threshold_map(s: str):
    """
    'blurry=0.7,dark=0.6' -> dict
    """
    if not s:
        return {}
    d = {}
    for part in s.split(","):
        part = part.strip()
        if not part: 
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --min-score-map entry: {part}")
        k, v = part.split("=", 1)
        d[k.strip()] = float(v.strip())
    return d


def main():
    ap = argparse.ArgumentParser(
        description="Stage or move flagged images into per-issue subfolders inside each image's directory."
    )
    ap.add_argument("csv", type=Path, help="issues.csv exported by Cleanlab/CleanVision")
    ap.add_argument("--root", type=Path, default=None, help="Optional dataset root to resolve relative paths.")
    ap.add_argument("--min-score", type=float, default=None, help="If set, treat rows with score >= min-score as issues (in addition to boolean flags).")
    ap.add_argument("--min-score-map", type=str, default=None, help="Per-issue overrides, e.g. 'blurry=0.7,dark=0.6'.")
    ap.add_argument("--only-bool", action="store_true", help="Only use boolean is_*_issue columns; ignore scores even if provided.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Path column always the first (unnamed) column
    path_col = df.columns[0]
    # If the first column is a numeric index (not a path), drop it and re-pick.
    if (path_col.startswith("Unnamed:") or path_col == "") and not looks_like_path(df[path_col]):
        if pd.api.types.is_integer_dtype(df[path_col]) and df[path_col].is_monotonic_increasing:
            df = df.drop(columns=[path_col])
            path_col = df.columns[0]
    # Final sanity: must look like a path.
    if not looks_like_path(df[path_col]):
        raise ValueError(f"First column '{path_col}' does not look like paths. "
                         f"Ensure the CSV’s first column contains file paths.")

    # Normalize path strings (strip spaces)
    df[path_col] = df[path_col].astype(str).str.strip()

    issue_cols = parse_issue_columns(df.columns)
    if not issue_cols:
        print("No issue columns found (expected pairs like *_score and is_*_issue).", file=sys.stderr)
        sys.exit(1)

    per_issue_thr = parse_threshold_map(args.min_score_map)

    total_rows = 0
    total_actions = 0
    missing_files = 0

    for idx, row in df.iterrows():
        p = Path(str(row[path_col]))
        if not p.is_absolute() and args.root:
            p = (args.root / p).resolve()
        elif not p.is_absolute():
            p = p.resolve()

        if not p.exists():
            missing_files += 1
            continue

        # Determine which issues apply to this row
        hits = []
        for issue, cols in issue_cols.items():
            hit = False
            score_val = None

            # 1) boolean flag
            if "flag" in cols and pd.notna(row[cols["flag"]]):
                try:
                    hit = bool(row[cols["flag"]])
                except Exception:
                    pass

            # 2) score threshold (if enabled)
            if not args.only_bool and ("score" in cols) and pd.notna(row[cols["score"]]):
                score_val = float(row[cols["score"]])
                thr = per_issue_thr.get(issue, args.min_score)
                if thr is not None and score_val >= thr:
                    hit = True

            if hit:
                hits.append((issue, score_val))

        if not hits:
            continue

        total_rows += 1
        parent = p.parent
        issues_root = parent / "_issues"

        # Choose primary issue bin: highest score first; if no scores, alphabetical
        primary_issue = sorted(hits, key=lambda x: (-(x[1] if x[1] is not None else -1), x[0]))[0][0]
        primary_dst = issues_root / primary_issue / p.name
        primary_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(primary_dst))
        total_actions += 1

        # For any additional issue bins, place hardlinks from the moved file
        other_bins = [issue for issue,_ in hits if issue != primary_issue]
        for issue in other_bins:
            extra_dst = issues_root / issue / primary_dst.name
            try:
                # always hardlink secondary bins
                extra_dst.parent.mkdir(parents=True, exist_ok=True)
                if extra_dst.exists():
                    return
                os.link(primary_dst, extra_dst)
            except Exception:
                shutil.copy2(primary_dst, extra_dst)
            total_actions += 1

    print(f"Done. Rows with ≥1 issue: {total_rows}")
    print(f"File operations performed: {total_actions}")
    if missing_files:
        print(f"Warning: {missing_files} files listed in CSV were missing on disk.", file=sys.stderr)

if __name__ == "__main__":
    main()
