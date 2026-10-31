#!/usr/bin/env python3
"""Independently recount two training-selected users from original SWEET.TV CSVs.

No production modules are imported. Selection is conditional on cached typed
training events, as requested; classification is recomputed from raw CSVs using
per-airing clipped-fragment endpoint sums, not a global viewing-interval union.
All source clocks remain naive. Integer seconds are a coordinate anchored at
1970-01-01, not evidence that source timestamps were recorded in UTC.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import tempfile
from time import perf_counter

import duckdb
import pyarrow.parquet as pq


TRAIN_END = 1_588_550_400
FORECAST_END = 1_595_808_000
EPOCH = datetime(1970, 1, 1)
LOG_ROLE = "dataset11-30week.csv"
EPG_ROLES = ("export_arh_10-19.csv", "export_arh_20-30.csv", "export_arh_31-42.csv")
KEYS = "user_id,channel_id,program_id,airing_start,airing_end"
INTEGER_PATTERN = r"[+-]?[0-9]+(\.0+)?"


def lit(value):
    return "'" + str(value).replace("'", "''") + "'"


def integer(column):
    return f"CASE WHEN regexp_full_match(trim({column}), {lit(INTEGER_PATTERN)}) THEN try_cast(trim({column}) AS BIGINT) END"


def wall(seconds):
    return (EPOCH + timedelta(seconds=int(seconds))).isoformat(sep=" ")


def encode(value):
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, (date, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def stat(path):
    value = path.stat()
    return {"bytes": value.st_size, "mtime_ns": value.st_mtime_ns}


class Queries:
    def __init__(self, connection):
        self.connection, self.records = connection, []

    def run(self, name, sql, parameters=None):
        started = perf_counter()
        cursor = self.connection.execute(sql, parameters or [])
        names = [column[0] for column in cursor.description] if cursor.description else []
        result = [dict(zip(names, row)) for row in cursor.fetchall()]
        self.records.append({"name": name, "sql": sql, "parameters": parameters or [],
                             "elapsed_seconds": perf_counter() - started, "rows_returned": len(result)})
        return result

    def one(self, name, sql, parameters=None):
        rows = self.run(name, sql, parameters)
        if len(rows) != 1:
            raise ValueError(f"{name}: expected one aggregate row")
        return rows[0]


def csv_source(path, columns):
    explicit = "{" + ",".join(f"{lit(name)}:'VARCHAR'" for name in columns) + "}"
    return (f"read_csv({lit(path)}, all_varchar=true, auto_detect=false, columns={explicit}, "
            "header=true, delim=',', quote='\"', escape='\"', encoding='utf-8', strict_mode=true, ignore_errors=false)")


def endpoint_sweep(q, fragments, prefix):
    q.run(prefix + "_endpoints", f"""CREATE TABLE {prefix}_points AS SELECT {KEYS},point,sum(delta) AS delta
        FROM (SELECT {KEYS},clip_start AS point,1 AS delta FROM {fragments}
              UNION ALL SELECT {KEYS},clip_end AS point,-1 AS delta FROM {fragments})
        GROUP BY {KEYS},point""")
    q.run(prefix + "_segments", f"""CREATE TABLE {prefix}_segments AS SELECT *,
        lead(point) OVER (PARTITION BY {KEYS} ORDER BY point) AS next_point,
        sum(delta) OVER (PARTITION BY {KEYS} ORDER BY point ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS active
        FROM {prefix}_points""")
    q.run(prefix + "_coverage", f"""CREATE TABLE {prefix}_coverage AS SELECT {KEYS},
        sum(CASE WHEN active>0 AND next_point IS NOT NULL THEN next_point-point ELSE 0 END) AS covered_seconds,
        min(active) AS minimum_active,max(active) AS maximum_active,
        max(active) FILTER (WHERE next_point IS NULL) AS final_active
        FROM {prefix}_segments GROUP BY {KEYS}""")


def self_check(q):
    # Expected lengths are stated before executing the independent SQL oracle.
    # First airing includes duplicate/overlapping fragments; the last has an
    # exact endpoint handoff. The two 60s repeats must not pool into one event.
    expected = {0: 80, 100: 79, 200: 60, 300: 60, 400: 80}
    q.run("selfcheck_fragments", """CREATE TABLE toy_fragments AS SELECT 'toy' AS user_id,'7' AS channel_id,
        11 AS program_id,a AS airing_start,a+100 AS airing_end,s AS clip_start,e AS clip_end
        FROM (VALUES (0,0,40),(0,20,60),(0,20,60),(0,80,100),(100,100,179),
             (200,200,260),(300,300,360),(400,400,440),(400,440,480)) v(a,s,e)""")
    endpoint_sweep(q, "toy_fragments", "toy")
    actual = q.run("selfcheck_results", "SELECT airing_start,covered_seconds,5*covered_seconds>=4*(airing_end-airing_start) AS qualifies FROM toy_coverage ORDER BY airing_start")
    if {row["airing_start"]: row["covered_seconds"] for row in actual} != expected:
        raise AssertionError("Independent endpoint-sweep smoke fixture failed")
    if [row["airing_start"] for row in actual if row["qualifies"]] != [0, 400]:
        raise AssertionError("Exact threshold/repeat-airing smoke fixture failed")
    return {"passed": True, "manual_expected_covered_seconds": expected, "qualifying_starts": [0, 400]}


def select_users(q, template_path, cache):
    reader = csv_source(template_path, ["user_id", "tv_show_id"])
    q.run("raw_template", f"CREATE TABLE template AS SELECT row_number() OVER () AS template_order,user_id FROM {reader}")
    template = q.run("template_strings", "SELECT user_id FROM template ORDER BY template_order")
    users = [row["user_id"] for row in template]
    if not users or len(set(users)) != len(users) or any(not isinstance(user, str) or not user for user in users):
        raise ValueError("Raw template must contain distinct nonempty string IDs")
    q.run("cached_events_view", f"CREATE VIEW cached_events AS SELECT user_id,tv_show_id,\"start\" AS start_s,\"end\" AS end_s,airing_id,coverage FROM read_parquet({lit(cache / 'events.parquet')})")
    q.run("cached_schedules_view", f"CREATE VIEW cached_schedules AS SELECT airing_id,channel_id,\"start\" AS start_s,\"end\" AS end_s,tv_show_id,quarantined FROM read_parquet({lit(cache / 'schedules.parquet')})")
    q.run("cached_training_counts_for_selection", """CREATE TABLE selection_counts AS SELECT e.user_id,e.tv_show_id,count(*) AS broadcasts
        FROM cached_events e JOIN template t USING(user_id) WHERE e.end_s<=? GROUP BY 1,2""", [TRAIN_END])
    descriptions = q.run("selection_descriptions", """SELECT t.user_id,coalesce(sum(c.broadcasts),0) AS total,
        coalesce(max(c.broadcasts),0) AS max_program_count,count(c.tv_show_id) AS distinct_programs
        FROM template t LEFT JOIN selection_counts c USING(user_id) GROUP BY t.user_id""")
    by_user = {row["user_id"]: row for row in descriptions}
    active = sorted((user for user in users if by_user[user]["total"] > 0), key=lambda user: (by_user[user]["total"], user))
    low, high = (len(active) // 4, max(len(active) // 4 + 2, 3 * len(active) // 4)) if len(active) >= 4 else (0, len(active))
    pool = active[low:high]
    pool.sort(key=lambda user: (Fraction(by_user[user]["max_program_count"], by_user[user]["total"]), user))
    if len(pool) >= 2:
        chosen = [pool[0], pool[-1]]
    else:
        chosen = pool[:]
        chosen += [user for user in users if user not in chosen][:2 - len(chosen)]
    if len(chosen) != 2:
        raise ValueError("Two distinct example users are required")
    details = {}
    for user in chosen:
        row = by_user[user]
        share = Fraction(row["max_program_count"], row["total"]) if row["total"] else Fraction(0)
        details[user] = {**row, "top_program_share": float(share), "exact_share": str(share),
                         "above_signed_int64": int(user) > 2**63 - 1 if user.isascii() and user.isdigit() else None}
    q.run("chosen_user_table", f"CREATE TABLE selected_users AS SELECT user_id FROM template WHERE user_id IN ({','.join('?' for _ in chosen)})", chosen)
    return {"selected_users": chosen, "template_users": len(users), "active_template_users": len(active),
            "middle_pool_slice": [low, high], "middle_pool_size": len(pool), "descriptions": details,
            "policy": "Training-only end<=1588550400; sort activity by (qualifying count,user string); middle 25%-75%; min/max exact max-program/total share, then user string",
            "selection_dependency": "Selection uses the typed cache for all template users; raw classification is independently checked only for the two selected users"}


def load_raw_subset(q, raw, mapping, source_audit):
    reader = csv_source(raw / mapping[LOG_ROLE], source_audit["files"][LOG_ROLE]["raw_columns"])
    q.run("raw_selected_logs", f"""CREATE TABLE raw_logs AS SELECT row_number() OVER () AS selected_scan_id,
        r.user_id,r.vsetv_id AS channel_id,r.start_time,r.stop_time,r.duraton,
        cast(epoch(strptime(r.start_time,'%Y-%m-%d %H:%M:%S')) AS BIGINT) AS start_s,
        cast(epoch(strptime(r.stop_time,'%Y-%m-%d %H:%M:%S')) AS BIGINT) AS end_s,
        {integer('r.duraton')} AS duration_raw_seconds
        FROM {reader} r JOIN selected_users u USING(user_id)""")
    logs = q.run("raw_log_counts", """SELECT user_id,count(*) AS raw_rows,min(start_s) AS first_start,max(end_s) AS last_stop,
        count(*) FILTER (WHERE end_s<=start_s) AS invalid_intervals,
        count(*) FILTER (WHERE duration_raw_seconds IS NULL) AS invalid_raw_duration_rows,
        count(*) FILTER (WHERE duration_raw_seconds!=end_s-start_s) AS duration_endpoint_mismatch_rows
        FROM raw_logs GROUP BY user_id ORDER BY user_id""")
    if any(row["invalid_intervals"] for row in logs):
        raise ValueError("Selected raw logs contain invalid intervals; do not silently repair")
    earliest = min(row["first_start"] for row in logs)
    latest = max(row["last_stop"] for row in logs)
    if latest > FORECAST_END:
        raise ValueError("Selected raw observations extend beyond the declared July 27 scope")
    parts = []
    for index, role in enumerate(EPG_ROLES):
        name = mapping[role]
        reader = csv_source(raw / name, source_audit["files"][role]["raw_columns"])
        q.run("raw_epg_" + str(index), f"""CREATE TABLE raw_epg_{index} AS WITH typed AS (
            SELECT channel_id,tv_show_title,{integer('tv_show_id')} AS program_id,
                {integer('duration')} AS duration_seconds,
                strptime(start_time,'%d.%m.%Y %H:%M:%S') AS start_wall
            FROM {reader})
            SELECT {lit(name)} AS source_file,channel_id,tv_show_title,program_id,duration_seconds,
                cast(epoch(start_wall) AS BIGINT) AS airing_start,
                cast(epoch(start_wall) AS BIGINT)+duration_seconds AS airing_end
            FROM typed WHERE program_id>0 AND cast(epoch(start_wall) AS BIGINT)<?
                AND cast(epoch(start_wall) AS BIGINT)+duration_seconds>?""", [FORECAST_END, earliest])
        parts.append(f"SELECT * FROM raw_epg_{index}")
    q.run("raw_epg_union", "CREATE VIEW raw_epg AS " + " UNION ALL ".join(parts))
    invalid = q.one("raw_positive_epg_interval_check", """SELECT count(*) AS applicable_positive_source_rows,
        count(*) FILTER (WHERE duration_seconds IS NULL OR duration_seconds<=0 OR channel_id IS NULL OR channel_id='') AS invalid_rows,
        min(airing_start) AS earliest_airing_start,max(airing_end) AS latest_airing_end
        FROM raw_epg""")
    if invalid["invalid_rows"]:
        raise ValueError("Applicable positive EPG has invalid intervals")
    q.run("raw_broadcast_identity", """CREATE TABLE raw_airings AS SELECT channel_id,program_id,airing_start,airing_end,
        min(tv_show_title) AS title,list_sort(list(DISTINCT source_file)) AS source_files,count(*) AS source_rows
        FROM raw_epg GROUP BY 1,2,3,4""")
    conflicts = q.one("assert_no_applicable_positive_epg_overlap", """WITH ordered AS (
        SELECT *,max(airing_end) OVER (PARTITION BY channel_id ORDER BY airing_start,airing_end,program_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS previous_max_end FROM raw_airings)
        SELECT count(*) AS unique_positive_airings,count(*) FILTER (WHERE airing_start<previous_max_end) AS overlapping_predecessor_rows,
            coalesce(sum(source_rows-1),0) AS duplicate_extra_source_rows FROM ordered""")
    if conflicts["overlapping_predecessor_rows"]:
        raise ValueError("Independent raw positive-EPG overlap assertion failed; no quarantine policy guessed")
    return {"logs": logs, "epg": invalid, "positive_epg_overlap_assertion": conflicts,
            "scope": "Positive airings starting before July 27 and ending after the first selected log; airing ends may cross July 27 and are not truncated"}


def reconstruct(q):
    q.run("raw_clipped_fragments", """CREATE TABLE raw_fragments AS SELECT l.user_id,a.channel_id,a.program_id,a.airing_start,a.airing_end,
        l.selected_scan_id,l.start_time,l.stop_time,l.duraton,l.duration_raw_seconds,
        greatest(l.start_s,a.airing_start) AS clip_start,least(l.end_s,a.airing_end) AS clip_end
        FROM raw_logs l JOIN raw_airings a ON l.channel_id=a.channel_id
            AND l.start_s<a.airing_end AND l.end_s>a.airing_start""")
    endpoint_sweep(q, "raw_fragments", "raw")
    integrity = q.one("endpoint_balance_checks", """SELECT count(*) AS matched_user_airings,
        count(*) FILTER (WHERE minimum_active<0 OR final_active!=0) AS unbalanced_endpoint_groups,
        count(*) FILTER (WHERE covered_seconds<0 OR covered_seconds>airing_end-airing_start) AS invalid_coverage_groups
        FROM raw_coverage""")
    if integrity["unbalanced_endpoint_groups"] or integrity["invalid_coverage_groups"]:
        raise AssertionError("Independent endpoint conservation failed")
    q.run("raw_qualifying_events", """CREATE TABLE raw_qualifying AS SELECT * FROM raw_coverage
        WHERE 5*covered_seconds>=4*(airing_end-airing_start)""")
    integrity.update(q.one("raw_reconstruction_dimensions", """SELECT (SELECT count(*) FROM raw_logs) AS selected_raw_log_rows,
        (SELECT count(*) FROM raw_fragments) AS clipped_fragment_rows,
        (SELECT count(*) FROM raw_points) AS grouped_endpoint_rows,
        (SELECT count(*) FROM raw_qualifying) AS qualifying_user_airings"""))
    return integrity


def compare(q, chosen):
    q.run("cache_selected_event_channels", """CREATE TABLE cached_selected AS SELECT e.user_id,s.channel_id,e.tv_show_id AS program_id,
        e.start_s AS airing_start,e.end_s AS airing_end,e.airing_id,e.coverage,
        s.airing_id IS NULL AS missing_schedule,
        e.start_s!=s.start_s OR e.end_s!=s.end_s OR e.tv_show_id!=s.tv_show_id AS inconsistent_schedule_fields,
        s.quarantined FROM cached_events e JOIN selected_users u USING(user_id)
        LEFT JOIN cached_schedules s USING(airing_id)""")
    mapping = q.one("cache_event_mapping_checks", """SELECT count(*) AS cache_selected_events,
        count(*) FILTER (WHERE missing_schedule) AS missing_schedule_rows,
        count(*) FILTER (WHERE inconsistent_schedule_fields) AS inconsistent_event_schedule_rows,
        count(*) FILTER (WHERE quarantined) AS events_on_quarantined_schedules,
        count(*)-count(DISTINCT (user_id,channel_id,program_id,airing_start,airing_end)) AS duplicate_event_rows
        FROM cached_selected""")
    q.run("raw_only_events", f"CREATE TABLE raw_only AS SELECT {KEYS} FROM raw_qualifying EXCEPT SELECT {KEYS} FROM cached_selected")
    q.run("cache_only_events", f"CREATE TABLE cache_only AS SELECT {KEYS} FROM cached_selected EXCEPT SELECT {KEYS} FROM raw_qualifying")
    differences = q.one("event_set_differences", "SELECT (SELECT count(*) FROM raw_only) AS raw_qualifying_missing_from_cache,(SELECT count(*) FROM cache_only) AS cached_events_not_qualifying_in_raw")
    differences["raw_only_examples"] = q.run("raw_only_examples", "SELECT * FROM raw_only ORDER BY user_id,airing_start,program_id LIMIT 10")
    differences["cache_only_examples"] = q.run("cache_only_examples", "SELECT * FROM cache_only ORDER BY user_id,airing_start,program_id LIMIT 10")
    coverage = q.one("cached_coverage_agreement", f"""SELECT count(*) AS common_events,
        count(*) FILTER (WHERE c.coverage IS NULL OR abs(c.coverage-r.covered_seconds::DOUBLE/(r.airing_end-r.airing_start))>1e-12) AS unequal_coverage_rows,
        max(abs(c.coverage-r.covered_seconds::DOUBLE/(r.airing_end-r.airing_start))) AS max_absolute_coverage_difference
        FROM cached_selected c JOIN raw_qualifying r USING({KEYS})""")
    users = {}
    for user in chosen:
        records = {}
        for side, table in (("raw", "raw_qualifying"), ("cache", "cached_selected")):
            totals = q.one(side + "_event_totals", f"""SELECT count(*) AS qualifying_events,
                count(*) FILTER (WHERE airing_end<=?) AS training_events,
                count(*) FILTER (WHERE airing_start>=? AND airing_end<=?) AS heldout_events,
                count(*) FILTER (WHERE airing_start<? AND airing_end>?) AS crosses_training_cutoff,
                count(*) FILTER (WHERE airing_start<? AND airing_end>?) AS crosses_forecast_end
                FROM {table} WHERE user_id=?""", [TRAIN_END, TRAIN_END, FORECAST_END, TRAIN_END, TRAIN_END, FORECAST_END, FORECAST_END, user])
            history = q.run(side + "_training_histogram", f"""SELECT program_id,count(*) AS broadcasts FROM {table}
                WHERE user_id=? AND airing_end<=? GROUP BY 1 ORDER BY broadcasts DESC,program_id""", [user, TRAIN_END])
            heldout = q.run(side + "_heldout_histogram", f"""SELECT program_id,count(*) AS broadcasts FROM {table}
                WHERE user_id=? AND airing_start>=? AND airing_end<=? GROUP BY 1 ORDER BY broadcasts DESC,program_id""", [user, TRAIN_END, FORECAST_END])
            records[side] = {**totals, "training_histogram": history, "heldout_histogram": heldout,
                             "heldout_top5": [row["program_id"] for row in heldout[:5]]}
        records["training_histograms_equal"] = records["raw"]["training_histogram"] == records["cache"]["training_histogram"]
        records["heldout_histograms_equal"] = records["raw"]["heldout_histogram"] == records["cache"]["heldout_histogram"]
        records["heldout_top5_equal"] = records["raw"]["heldout_top5"] == records["cache"]["heldout_top5"]
        users[user] = records
    success = (all(mapping[key] == 0 for key in ("missing_schedule_rows", "inconsistent_event_schedule_rows", "events_on_quarantined_schedules", "duplicate_event_rows"))
               and differences["raw_qualifying_missing_from_cache"] == differences["cached_events_not_qualifying_in_raw"] == 0
               and coverage["unequal_coverage_rows"] == 0
               and all(row["training_histograms_equal"] and row["heldout_histograms_equal"] and row["heldout_top5_equal"] for row in users.values()))
    return {"all_comparisons_agree": success, "cache_mapping_checks": mapping, "event_set_differences": differences,
            "coverage_comparison": coverage, "coverage_float_tolerance": 1e-12, "users": users}


def hand_example(q):
    examples = q.run("bounded_hand_check_candidate", f"""WITH fragments AS (
        SELECT {KEYS},count(*) AS fragment_count,sum(clip_end-clip_start) AS summed_fragment_seconds
        FROM raw_fragments GROUP BY {KEYS})
        SELECT r.*,f.fragment_count,f.summed_fragment_seconds,a.title,a.source_files
        FROM raw_qualifying r JOIN fragments f USING({KEYS})
        JOIN raw_airings a USING(channel_id,program_id,airing_start,airing_end)
        WHERE f.fragment_count BETWEEN 1 AND 3 AND r.covered_seconds<r.airing_end-r.airing_start
        ORDER BY CASE WHEN f.fragment_count>=2 THEN 0 ELSE 1 END,
            CASE WHEN f.summed_fragment_seconds>r.covered_seconds THEN 0 ELSE 1 END,
            r.airing_start,r.user_id,r.program_id LIMIT 1""")
    if not examples:
        return {"available": False, "personal_verification_performed": False}
    row = examples[0]
    values = [row["user_id"], row["channel_id"], row["program_id"], row["airing_start"], row["airing_end"]]
    predicate = "user_id=? AND channel_id=? AND program_id=? AND airing_start=? AND airing_end=?"
    fragments = q.run("hand_check_raw_log_fragments", f"""SELECT selected_scan_id,start_time AS raw_log_start,stop_time AS raw_log_stop,
        duraton AS raw_duraton,clip_start,clip_end,clip_end-clip_start AS clipped_seconds FROM raw_fragments
        WHERE {predicate} ORDER BY clip_start,clip_end,selected_scan_id""", values)
    segments = q.run("hand_check_endpoint_segments", f"""SELECT point,next_point,active,next_point-point AS seconds
        FROM raw_segments WHERE {predicate} AND next_point IS NOT NULL ORDER BY point""", values)
    for fragment in fragments:
        fragment["clipped_start_wall"], fragment["clipped_end_wall"] = wall(fragment["clip_start"]), wall(fragment["clip_end"])
    for segment in segments:
        segment["start_wall"], segment["end_wall"] = wall(segment["point"]), wall(segment["next_point"])
    duration = row["airing_end"] - row["airing_start"]
    return {"available": True, "user_id": row["user_id"], "channel_id": row["channel_id"], "program_id": row["program_id"],
            "title": row["title"], "source_files": row["source_files"],
            "broadcast_start_wall": wall(row["airing_start"]), "broadcast_end_wall": wall(row["airing_end"]),
            "scheduled_duration_seconds": duration, "clipped_fragments": fragments, "endpoint_segments": segments,
            "covered_union_seconds": row["covered_seconds"], "coverage_fraction": str(Fraction(row["covered_seconds"], duration)),
            "coverage_ratio": row["covered_seconds"] / duration,
            "threshold_products": {"5_times_covered": 5 * row["covered_seconds"], "4_times_scheduled": 4 * duration},
            "personal_verification_performed": False,
            "instruction": "Prepared for a later user-performed check; sum endpoint-segment lengths only where active>0. The row scan ID is within the selected scan, not an original CSV physical line number."}


def report(data):
    selected = data["selection"]
    comparison = data["comparison"]
    lines = ["# Independent raw-data recount of two primary example users", "",
             f"Generated: {data['generated_at_local']}.", "",
             f"**Raw/cache agreement: {comparison['all_comparisons_agree']}.** This compares an independent reconstruction with the existing normalized cache, not organizer-private targets or model quality.", "",
             "## Independence and clock contract", "",
             "No production parsing/counting/ranking helper was imported. The selection rule was read as a contract from `select_example_users`; the two IDs were independently derived from typed cached events and the raw template. All classification and broadcast counts for those users were then recomputed from actual CSVs.",
             "Both sources are treated as one timezone-unknown naive wall-clock coordinate. Numeric seconds subtract naive 1970-01-01; they are not a claim that source clocks were UTC. Raw `duraton` is audited but never repairs endpoints. Primary training ends at 2020-05-04 00:00:00 (1588550400); labels require whole airings inside [2020-05-04,2020-07-27).", "",
             "No cache, raw, production, model, existing report or test was edited. The running benchmark's output directory was not read or polled. Temporary DuckDB spill files were confined to the approved runtime temporary directory, not a data/cache directory.", "",
             "## Training-only user selection", "",
             f"Raw template users: **{selected['template_users']}**; active before cutoff: **{selected['active_template_users']}**. Middle activity slice `{selected['middle_pool_slice']}` contains **{selected['middle_pool_size']}** users. Ordering uses total training broadcast count then the exact user string, followed by min/max top-program share and string ties. Shares are compared as exact fractions.", "",
             "| Selected user | Training broadcasts (selection cache) | Distinct programs | Max program count / total |", "| --- | ---: | ---: | --- |"]
    for user in selected["selected_users"]:
        row = selected["descriptions"][user]
        lines.append(f"| `{user}` | {row['total']} | {row['distinct_programs']} | {row['max_program_count']}/{row['total']} = {row['exact_share']} |")
    lines.extend(["", "Selection depends on the cached training counts for the full template. This audit does not independently reconstruct all users to prove that ordering; its raw-data verification is restricted to the chosen two.", "",
                  "## Independent per-airing reconstruction", "",
                  "Positive raw EPG rows are parsed with `%d.%m.%Y %H:%M:%S` and whole-second duration; zero IDs are excluded first. Exact broadcast identity is `(channel,start,end,program_id)`. Historical scope includes broadcasts crossing the July 27 boundary without truncating their scheduled denominator.",
                  f"The applicable positive-EPG assertion found **{data['raw_scope']['positive_epg_overlap_assertion']['overlapping_predecessor_rows']}** overlaps and **{data['raw_scope']['positive_epg_overlap_assertion']['duplicate_extra_source_rows']}** extra exact rows before assuming nonconflicting airings.",
                  "Each raw log is intersected with each applicable same-channel airing. Its clipped start contributes +1 and clipped end -1. Simultaneous endpoints are grouped, active counts are cumulatively summed **within each user/airing**, and disjoint segment lengths count only when active>0. Eligibility uses exact `5*covered >= 4*scheduled_seconds`, once per user/broadcast. No global log-interval union or production sweep is reused.", "",
                  "| User | Raw log rows | All qualifying events raw/cache | Training events raw/cache | Fully contained held-out events raw/cache |", "| --- | ---: | --- | --- | --- |"])
    raw_logs = {row["user_id"]: row for row in data["raw_scope"]["logs"]}
    for user in selected["selected_users"]:
        raw, cache = comparison["users"][user]["raw"], comparison["users"][user]["cache"]
        lines.append(f"| `{user}` | {raw_logs[user]['raw_rows']} | {raw['qualifying_events']}/{cache['qualifying_events']} | {raw['training_events']}/{cache['training_events']} | {raw['heldout_events']}/{cache['heldout_events']} |")
    differences = comparison["event_set_differences"]
    lines.extend(["", f"Full canonical event-set differences: raw-only **{differences['raw_qualifying_missing_from_cache']}**, cache-only **{differences['cached_events_not_qualifying_in_raw']}**. Cache airing IDs were mapped to cached schedule channels and event/schedule fields checked, rather than reproducing the production ID hash.",
                  f"Coverage differences above 1e-12: **{comparison['coverage_comparison']['unequal_coverage_rows']}**; maximum absolute difference: `{comparison['coverage_comparison']['max_absolute_coverage_difference']}`. Exact integer thresholding was used for the raw oracle; tolerance applies only to comparison with cached floating coverage.", "",
                  "## Primary histories and labels", "",
                  "The JSON contains complete raw and cache training/held-out frequency histograms for each selected user. Top-five labels use broadcast frequency descending then numeric raw program ID ascending, not seconds or display titles.", ""])
    for user in selected["selected_users"]:
        row = comparison["users"][user]
        lines.extend([f"- `{user}`: training histogram equal **{row['training_histograms_equal']}**; held-out histogram equal **{row['heldout_histograms_equal']}**; top five `{row['raw']['heldout_top5']}`; cache top-five equal **{row['heldout_top5_equal']}**."])
    example = data["hand_verification_candidate"]
    lines.extend(["", "## Actual bounded hand-verification candidate", "", "This is evidence prepared for a later human check, **not a claim of user/personal verification**.", ""])
    if example["available"]:
        lines.extend([f"User `{example['user_id']}`, channel `{example['channel_id']}`, raw program ID `{example['program_id']}`; title: {example['title']}.",
                      f"Source EPG: `{example['source_files']}`; broadcast `{example['broadcast_start_wall']}` to `{example['broadcast_end_wall']}`, **{example['scheduled_duration_seconds']} seconds**.", "",
                      "| Raw log start | Raw log stop | Raw duraton | Clipped start | Clipped end | Clipped seconds |", "| --- | --- | ---: | --- | --- | ---: |"])
        for fragment in example["clipped_fragments"]:
            lines.append(f"| {fragment['raw_log_start']} | {fragment['raw_log_stop']} | {fragment['raw_duraton']} | {fragment['clipped_start_wall']} | {fragment['clipped_end_wall']} | {fragment['clipped_seconds']} |")
        lines.extend(["", f"Union coverage is **{example['covered_union_seconds']} / {example['scheduled_duration_seconds']} = {example['coverage_fraction']}** ({example['coverage_ratio']:.8f}). Exact comparison: `{example['threshold_products']['5_times_covered']} >= {example['threshold_products']['4_times_scheduled']}`. Endpoint segments/active counts are in JSON so overlaps are not double-counted."])
    else:
        lines.append("No qualifying incomplete-coverage airing with 1–3 clipped fragments was found; none is fabricated.")
    lines.extend(["", "## Checks, timing and limitations", "",
                  f"The manual SQL self-check passed (expected coverage 80,79,60,60,80 seconds; only first/last qualify). Actual raw recount used **{data['reconstruction']['clipped_fragment_rows']}** clipped fragments across **{data['reconstruction']['matched_user_airings']}** matched user-airings, with **{data['reconstruction']['unbalanced_endpoint_groups']}** endpoint-balance errors and **{data['reconstruction']['invalid_coverage_groups']}** impossible coverage totals.",
                  f"Executed **{len(data['executed_queries'])}** timed SQL statements in **{data['execution']['elapsed_seconds']:.3f} seconds** overall; DuckDB {data['execution']['duckdb_version']}, four threads / 4GB engine buffer limit. SQL and parameters are recorded in JSON.", "", "```sh", ".venv/bin/python -B tools/recount_official_users.py", "```", "",
                  "Inputs are anchored by manifests/audit metadata, schemas and before/after size/mtime checks. Existing large-file hashes are attributed, not freshly rehashed. Only the new profiler script and output files are created. Source timezone, hidden organizer fragment/conflict/tie policies and target fidelity remain unknown. Agreement establishes consistency with this declared reconstruction policy for two users, not official quality or a returned Kaggle score.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/official-v1"))
    parser.add_argument("--source-audit", type=Path, default=Path("results/official-preparation-001/source_audit.json"))
    parser.add_argument("--file-map", type=Path, default=Path("configs/source-files.json"))
    parser.add_argument("--temp-parent", type=Path, default=Path("../../.opencode/runtime/tmp/opencode"))
    parser.add_argument("--out", type=Path, default=Path("results/official-raw-recount.json"))
    parser.add_argument("--report", type=Path, default=Path("reviews/official-raw-recount.md"))
    args = parser.parse_args()
    if args.out.exists() or args.report.exists():
        raise FileExistsError("Recount/report outputs must be new")
    started = perf_counter()
    mapping = json.loads(args.file_map.read_text())
    audit = json.loads(args.source_audit.read_text())
    if mapping != audit["file_names"]:
        raise ValueError("Explicit source mapping differs from the preparation audit")
    cache_manifest = json.loads((args.cache / "manifest.json").read_text())
    paths = [args.raw_dir / name for name in mapping.values()]
    paths += [args.cache / name for name in ("events.parquet", "schedules.parquet", "manifest.json", "audit.json", "template_users.json")]
    paths += [args.source_audit, args.file_map]
    before = {str(path): stat(path) for path in paths}
    schemas = {}
    for name in ("events.parquet", "schedules.parquet"):
        parquet = pq.ParquetFile(args.cache / name)
        schemas[name] = {"rows": parquet.metadata.num_rows, "schema": str(parquet.schema_arrow),
                         "declared_sha256": cache_manifest["files"][name]["sha256"], "fresh_hash_computed": False}
    with tempfile.TemporaryDirectory(prefix="official-two-user-recount-", dir=args.temp_parent) as temporary:
        connection = duckdb.connect()
        q = Queries(connection)
        try:
            q.run("threads", "SET threads=4")
            q.run("memory", "SET memory_limit='4GB'")
            q.run("spill", "SET temp_directory=" + lit(Path(temporary) / "spill"))
            q.run("order", "SET preserve_insertion_order=true")
            smoke = self_check(q)
            selection = select_users(q, args.raw_dir / mapping["submission.csv"], args.cache)
            print("Independently selected: " + ", ".join(selection["selected_users"]), file=sys.stderr, flush=True)
            raw_scope = load_raw_subset(q, args.raw_dir, mapping, audit)
            reconstruction = reconstruct(q)
            comparison = compare(q, selection["selected_users"])
            example = hand_example(q)
        finally:
            connection.close()
    after = {str(path): stat(path) for path in paths}
    if before != after:
        raise RuntimeError("Raw/cache/audit input changed during the independent recount")
    data = {"generated_at_local": datetime.now().astimezone().isoformat(),
            "production_imports": False, "benchmark_outputs_consulted": False,
            "raw_or_cache_modified": False, "personal_verification_performed": False,
            "clock_policy": "Common naive wall-clock coordinate; numeric naive epoch anchor only, not a source-UTC assertion; no offset or DST conversion",
            "primary": {"train_end_anchor_seconds": TRAIN_END, "forecast_start_anchor_seconds": TRAIN_END,
                        "forecast_end_anchor_seconds": FORECAST_END, "train_end_wall": wall(TRAIN_END), "forecast_end_wall": wall(FORECAST_END)},
            "profiler_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_stat_before": before, "input_stat_after": after, "inputs_stat_unchanged": True,
            "source_sha256_declared_in_preparation_audit": {row["source_filename"]: row["sha256"] for row in audit["files"].values()},
            "cache_schemas": schemas,
            "small_input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.file_map, args.source_audit, args.cache / "manifest.json")},
            "oracle_self_check": smoke, "selection": selection, "raw_scope": raw_scope,
            "reconstruction": reconstruction, "comparison": comparison, "hand_verification_candidate": example,
            "execution": {"duckdb_version": duckdb.__version__, "python_version": platform.python_version(),
                          "threads": 4, "memory_limit": "4GB", "temporary_spill_parent": str(args.temp_parent),
                          "temporary_files_removed": True, "elapsed_seconds": perf_counter() - started,
                          "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if sys.platform == "darwin" else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024},
            "executed_queries": q.records}
    data = json.loads(json.dumps(data, default=encode, ensure_ascii=False, allow_nan=False))
    markdown = report(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    with args.report.open("x", encoding="utf-8") as stream:
        stream.write(markdown)
    print(json.dumps({"selected_users": selection["selected_users"], "all_comparisons_agree": comparison["all_comparisons_agree"],
                      "raw_qualifying_events": reconstruction["qualifying_user_airings"], "sql_statements": len(q.records),
                      "hand_example_available": example["available"], "elapsed_seconds": data["execution"]["elapsed_seconds"]}, indent=2))
    return 0 if comparison["all_comparisons_agree"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
