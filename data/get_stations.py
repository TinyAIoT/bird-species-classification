import argparse
from pathlib import Path
import requests
import pandas as pd


BASE_URL = "https://wiediversistmeingarten.org/api"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save a list of all Birdiary stations as a csv.")
    parser.add_argument("--workdir", type=str, default="./", help="Path to local data directory.")
    parser.add_argument("--user-agent", type=str, default="", help="User agent for requests.")
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)


    stations_url = f"{BASE_URL}/station"


    if args.user_agent:
        # Attach custom header with User-Agent
        response = requests.get(url=stations_url, headers={"User-Agent": args.user_agent})
    else:
        response = requests.get(url=stations_url)
    stations_data = response.json()

    station_list = []

    for station in stations_data:
        last_movement = station.get("lastMovement")
        station_list.append({
            "station_id": station["station_id"],
            "name": station["name"],
            "lat": station["location"]["lat"],
            "lng": station["location"]["lng"],
            # "sensebox_id": station.get("sensebox_id", "")
            "last_movement_date": last_movement.get("createdAt") if last_movement else None,
        })

    stations_df = pd.DataFrame(station_list)
    output_path = workdir / "station_ids.csv"
    stations_df.to_csv(output_path, index=False)
    print(f"Saved {len(stations_df.index)} stations to {output_path}.")