import argparse
from pathlib import Path
import time
import random
import requests
import pandas as pd


BASE_URL = "https://wiediversistmeingarten.org/api"
EXCLUDED_VALIDATIONS = [ # Manually crafted list of existing invalid/incorrect class labels
        "cyanocitta cristata", "poecile atricapillus", "sitta canadensis", "melospiza lincolnii", 
        "molothrus ater", "thryomanes bewickii", "thryothorus ludovicianus", "poecile carolinensis",
        "ardea herodias", "anthornis melanura", "hemiphaga novaeseelandiae", "seiurus aurocapilla",
        "egretta novaehollandiae", "alopochen aegyptiaca", "sitta pygmaea", "hirundo rustica", 
        "corvus corax", "accipiter nisus"
    ]


def process_movements(station_id, mov, validated_only):
    """Extracts the validation(s) and predictions from a movement"""
    validations = mov.get("validation", {}).get("validations", [])
    if validated_only and not validations:
        return None

    names = set()
    for val in validations:
        name = val.get("latinName", "").strip().lower()
        if not name or name == "none":
            # Videos without a bird not needed
            continue        

        if val.get("germanName", "") == "" and name != "homo sapiens":
            # If german name is empty the latin name contains an non specified species
            # station_id/mov_id in print for easy copy-paste website verification
            print(f"[SKIP INFO] {station_id}/{mov['mov_id']}, latinName of empty germanName: {name}")
            continue

        if name in EXCLUDED_VALIDATIONS or "familie" in name:
            continue
        names.add(name)

    if validated_only and not names:
        # Skip if no qualifying validations found and validated_only True
        return None

    detections = mov.get("detections", {})
    predictions = set()
    for det in detections:
        name = det.get("latinName", "").strip().lower()
        score = det.get("score")
        if name and score is not None:
            predictions.add((name, round(score, 2)))

    environment = mov.get("environment", {})

    return {
        "station_id": station_id,
        "mov_id": mov["mov_id"],
        "start_date": mov["start_date"],
        "weight": mov.get("weight", ""),
        # "temperature": environment.get("temperature", ""),
        # "humidity": environment.get("humidity", ""),
        # "airpressure": environment.get("airpressure", ""),
        # "illuminance": environment.get("illuminance", ""),
        "predictions": "; ".join(str(t) for t in sorted(predictions, key=lambda x: x[1], reverse=True)),
        "validations": "; ".join(str(t) for t in sorted(names)),
        "video_link": mov["video"],
    }


def fetch_with_retries(session, url, max_retries=5, base_delay=1):
    """Fetch with retries using an existing session"""
    # I had issues sometimes with unresponsiveness from many requests
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                return response
            else:
                print(f"[WARN] Status {response.status_code} for {url}")
        except requests.exceptions.RequestException as e:
            print(f"[WARN] Attempt {attempt + 1} failed: {e}")
        time.sleep(base_delay * (2**attempt) + random.uniform(0, 1))
    return None


def process_station(session, row, validated_only, output_csv_path, number_movements):
    """Go through the movements of a station and save incrementally to csv."""
    station_id = row["station_id"]
    station_name = str(row["name"])
    #print(f"[INFO] Fetching movements for station {station_name} ({station_id})")

    movements_url = f"{BASE_URL}/movement/{station_id}"
    if number_movements:
        movements_url += f"?movements={number_movements}"

    response = fetch_with_retries(session, movements_url)
    if response is None:
        print(f"[EXCEPTION] Station {station_id} failed after retries")
        return 0

    try:
        movements = response.json()
        if not movements:
            # Skip if the station has no movements                
            print(f"[INFO] No movements for station: {station_name}")
            return 0

        # Go through all of the movements of the station
        num_movs = 0
        for mov in movements:
            processed = process_movements(station_id, mov, validated_only)
            if processed:
                df = pd.DataFrame([processed])
                df.to_csv(output_csv_path, mode="a", header=not Path(output_csv_path).exists(), index=False)
                num_movs += 1

        if num_movs > 0:
            print(f"[OK] Collected {num_movs} movement(s) for: {station_name}")
        else:
            print(f"[INFO] No movements for station: {station_name}")
        return num_movs
    except Exception as e:
        print(f"[EXCEPTION] Error processing station {station_id}: {e}")



def main():
    parser = argparse.ArgumentParser(description="Fetch movements for stations and save to CSV.")
    parser.add_argument("--workdir", type=str, default="./", help="Path to local data directory.")
    parser.add_argument("--user-agent", type=str, default="", help="User agent for requests.")
    parser.add_argument("--validated-only", action=argparse.BooleanOptionalAction, 
        default=True, help="Only include movements with at least one human validation.")
    parser.add_argument("--number-movements", type=int, help="Number of movements to retrieve per station (API limit).")
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    stations_csv = workdir / "station_ids.csv"
    if not stations_csv.exists():
        raise FileNotFoundError(f"Missing {stations_csv}!")
    stations_df = pd.read_csv(stations_csv)

    # Path for collected data
    output_csv_path = workdir / "movements.csv"

    with requests.Session() as session:
        if args.user_agent:
            # Attach custom header with User-Agent
            session.headers.update({"User-Agent": args.user_agent})

        # Adjust connection pooling
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        sum_movs = 0
        for _, row in stations_df.iterrows():
            num_movs = process_station(session, row, args.validated_only, output_csv_path, args.number_movements)
            sum_movs += num_movs
            # Delay between stations
            # time.sleep(random.uniform(1.0, 2.5))

    print(f"[OK] Saved total {sum_movs} movements to {output_csv_path}")

if __name__ == "__main__":
    main()
