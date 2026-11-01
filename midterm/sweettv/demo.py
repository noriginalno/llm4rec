"""Clearly synthetic local integration data. Never evidence of Kaggle quality."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random


def make_demo(raw_dir: Path, seed: int = 42, user_count: int = 72) -> dict:
    if raw_dir.exists():
        raise FileExistsError(raw_dir)
    raw_dir.mkdir(parents=True)
    rng = random.Random(seed)
    users = [str(8_200_000_000_000_000_001 + i * 101) for i in range(user_count)]
    metadata_fields = ["channel_id", "channel_title", "start_time", "duration", "tv_show_id", "tv_show_title", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3", "year_of_production", "director", "actors"]
    schedules = {"export_arh_10-19.csv": [], "export_arh_20-30.csv": [], "export_arh_31-42.csv": []}
    logs = []
    categories = ["Документальные", "Спорт", "Развлечения"]
    for week in range(10, 43):
        monday = datetime.fromisocalendar(2020, week, 1).replace(tzinfo=timezone.utc)
        for item in range(1, 40):
            if item > 36 and week < 23:
                continue
            channel = (item - 1) // 13 + 1
            category = (item - 1) // 13
            for day in (0, 2, 4):
                start = monday + timedelta(days=day, hours=12, minutes=30 * ((item - 1) % 13))
                end = start + timedelta(seconds=1200)
                row = [str(channel), f"Synthetic channel {channel}", start.isoformat(), "1200", str(item), f"{categories[category]} передача {item}", categories[category], categories[category], "", "", "2020", "", ""]
                key = "export_arh_10-19.csv" if week <= 19 else "export_arh_20-30.csv" if week <= 30 else "export_arh_31-42.csv"
                schedules[key].append(row)
                if not 11 <= week <= 30:
                    continue
                for user_number, user in enumerate(users):
                    if user_number == user_count - 1 and week < 19:
                        continue
                    preferred_category = user_number % 3
                    preferred = {1 + preferred_category * 13 + (user_number // 3 + offset) % 12 for offset in range(7)}
                    if item not in preferred and not (item > 36 and preferred_category == 2 and rng.random() < 0.25):
                        continue
                    if rng.random() < 0.15:
                        continue
                    fraction = 0.85 if rng.random() < 0.92 else 0.6
                    watch_end = start + timedelta(seconds=int(1200 * fraction))
                    if (user_number + item + week) % 11 == 0:
                        middle = start + timedelta(seconds=480)
                        logs.append([user, str(channel), start.isoformat(), middle.isoformat(), "480"])
                        logs.append([user, str(channel), middle.isoformat(), watch_end.isoformat(), str(int((watch_end-middle).total_seconds()))])
                    else:
                        logs.append([user, str(channel), start.isoformat(), watch_end.isoformat(), str(int((watch_end-start).total_seconds()))])
    def write(name: str, header: list[str], rows: list[list[str]]) -> None:
        with (raw_dir / name).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
    for name, rows in schedules.items():
        write(name, metadata_fields, rows)
    write("dataset11-30week.csv", ["user_id", "vsetv_id", "start_time", "end_time", "duration"], logs)
    write("submission.csv", ["user_id", "tv_show_id"], [[u, "0 0 0 0 0"] for u in users])
    manifest = {"synthetic": True, "not_kaggle_data": True, "seed": seed, "users": user_count, "log_rows": len(logs)}
    (raw_dir / "SYNTHETIC.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
