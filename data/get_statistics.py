import argparse
from pathlib import Path
import time
import random
import requests
import pandas as pd

BASE_URL = "https://wiediversistmeingarten.org/api"


def fetch_with_retries(session, url, max_retries=5, base_delay=1):
    """Fetch with retries using an existing session."""
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                return response
            else:
                print(f"[WARN] Status {response.status_code} for {url}")
        except requests.exceptions.RequestException as e:
            print(f"[WARN] Attempt {attempt + 1} failed for {url}: {e}")
        time.sleep(base_delay * (2**attempt) + random.uniform(0, 1))
    return None


def process_station_statistics(session, row):
    """Fetch /statistics/{station_id} and return a flat dict for CSV."""
    station_id = row["station_id"]
    station_name = str(row.get("name", station_id))

    stats_url = f"{BASE_URL}/statistics/{station_id}"
    response = fetch_with_retries(session, stats_url)
    if response is None:
        print(f"[EXCEPTION] Station {station_id} statistics failed after retries")
        return None

    try:
        stats = response.json()
    except ValueError as e:
        print(f"[EXCEPTION] JSON decode error for station {station_id}: {e}")
        return None

    # Basic numeric fields
    number_of_movements = stats.get("numberOfMovements", 0)
    number_of_detections = stats.get("numberOfDetections", 0)
    number_of_validated_birds = stats.get("numberOfValidatedBirds", 0)

    # Validated birds: dict keyed by latin name
    validated = stats.get("validatedBirds", {}) or {}
    validated_tuples = []

    # Structure:
    # "validatedBirds": {
    #   "Parus major": { "sum": 1413, ... },
    #   ...
    # }
    for latin_name, info in validated.items():
        sum_count = info.get("sum")
        if sum_count is None:
            continue
        validated_tuples.append((latin_name, sum_count))

    # Sort by descending count to have deterministic ordering
    validated_tuples.sort(key=lambda x: x[1], reverse=True)

    # Stringify list of tuples for CSV, e.g. "(Parus major, 1413); (Cyanistes caeruleus, 346)"
    validated_str = "; ".join(
        f"({latin}, {count})" for latin, count in validated_tuples
    )

    result = {
        "station_id": stats.get("station_id", station_id),
        "numberOfMovements": number_of_movements,
        "numberOfDetections": number_of_detections,
        "numberOfValidatedBirds": number_of_validated_birds,
        "validatedBirds": validated_str,
    }

    print(f"[OK] {station_name} ({station_id}): movements={number_of_movements}, detections={number_of_detections}, validated_birds={number_of_validated_birds}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch station statistics and save to CSV.")
    parser.add_argument("--workdir", type=str, default="./", help="Path to local data directory (must contain station_ids.csv).",)
    parser.add_argument("--user-agent", type=str, default="", help="User agent for requests.")
    parser.add_argument("--output", type=str, default="station_statistics.csv", help="Output CSV filename (relative to workdir).")
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    stations_csv = workdir / "station_ids.csv"
    if not stations_csv.exists():
        raise FileNotFoundError(f"Missing {stations_csv}!")

    stations_df = pd.read_csv(stations_csv)

    output_csv_path = workdir / args.output

    with requests.Session() as session:
        if args.user_agent:
            session.headers.update({"User-Agent": args.user_agent})

        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # Iterate stations and append rows incrementally
        first_write = True
        for _, row in stations_df.iterrows():
            result = process_station_statistics(session, row)
            if result is None:
                continue

            df = pd.DataFrame([result])
            df.to_csv(
                output_csv_path,
                mode="a",
                header=first_write,
                index=False,
            )
            first_write = False

    print(f"[OK] Saved station statistics to {output_csv_path}")


if __name__ == "__main__":
    main()
