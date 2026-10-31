#!/usr/bin/env python3
"""Independent bounded DuckDB source profile; no production parser or reconstruction.

All CSV columns are read as VARCHAR with strict UTF-8 decoding. User identifiers
remain exact strings. Timestamp parsing produces timezone-naive wall-clock
TIMESTAMP values, not UTC instants. Source SHA-256 values are attributed to the
existing acquisition manifest, not recomputed over the large CSVs.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import tempfile
from time import perf_counter

import duckdb


LOG_ROLE = "dataset11-30week.csv"
EPG_ROLES = ("export_arh_10-19.csv", "export_arh_20-30.csv", "export_arh_31-42.csv")
TEMPLATE_ROLE = "submission.csv"
INTEGER_TEXT = r"[+-]?[0-9]+(\.0+)?"
NULL_WORDS = "('', 'null', 'none', 'nan', 'na', 'n/a', '\\n')"


def literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def missing(column: str) -> str:
    return f"({column} IS NULL OR lower(trim({column})) IN {NULL_WORDS})"


def present(column: str) -> str:
    return f"NOT {missing(column)}"


def integer(column: str) -> str:
    return (f"CASE WHEN regexp_full_match(trim({column}), {literal(INTEGER_TEXT)}) "
            f"THEN try_cast(trim({column}) AS HUGEINT) END")


def json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()
    if isinstance(value, (Decimal, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


class Queries:
    def __init__(self, connection):
        self.connection = connection
        self.records = []

    def run(self, name, sql, parameters=None):
        started = perf_counter()
        cursor = self.connection.execute(sql, parameters or [])
        columns = [column[0] for column in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        self.records.append({"name": name, "sql": sql, "parameters": parameters or [],
                             "elapsed_seconds": perf_counter() - started, "rows_returned": len(rows)})
        return rows

    def one(self, name, sql, parameters=None):
        rows = self.run(name, sql, parameters)
        if len(rows) != 1:
            raise ValueError(f"{name}: expected one aggregate row, got {len(rows)}")
        return rows[0]


def csv_reader(path, headers):
    # Header names are independently checked against the prior bounded sample;
    # explicit types/names and CSV dialect avoid header/type guessing in DuckDB.
    columns = "{" + ",".join(f"{literal(name)}:'VARCHAR'" for name in headers) + "}"
    return (f"read_csv({literal(str(path))}, header=true, delim=',', quote='\"', escape='\"', "
            f"auto_detect=false, columns={columns}, all_varchar=true, encoding='utf-8', "
            "strict_mode=true, ignore_errors=false)")


def load_sources(q, raw, mapping, sample, manifest):
    sources = {}
    for role in (TEMPLATE_ROLE, LOG_ROLE, *EPG_ROLES):
        name = mapping[role]
        path = raw / name
        stat = path.stat()
        with path.open(encoding="utf-8-sig", errors="strict", newline="") as stream:
            headers = next(csv.reader(stream, strict=True))
        if headers != sample["files"][name]["columns"] or len(set(headers)) != len(headers):
            raise ValueError(f"{name}: actual headers differ from the supplied bounded sample")
        declared = manifest["files"][name]
        if stat.st_size != declared["bytes"]:
            raise ValueError(f"{name}: byte size differs from the acquisition manifest")
        reader = csv_reader(path, headers)
        if role == TEMPLATE_ROLE:
            table, sql = "template", f"SELECT * FROM {reader}"
        elif role == LOG_ROLE:
            table = "logs"
            sql = f"""SELECT *,
                try_strptime(start_time, '%Y-%m-%d %H:%M:%S') AS start_wall,
                try_strptime(stop_time, '%Y-%m-%d %H:%M:%S') AS stop_wall,
                {integer('duraton')} AS duration_integer,
                try_cast(duraton AS DOUBLE) AS duration_number
                FROM {reader}"""
        else:
            table = f"epg_{EPG_ROLES.index(role)}"
            sql = f"""WITH typed AS (SELECT *,
                try_strptime(start_time, '%d.%m.%Y %H:%M:%S') AS start_wall,
                {integer('duration')} AS duration_integer,
                try_cast(duration AS DOUBLE) AS duration_number,
                {integer('tv_show_id')} AS program_id
                FROM {reader})
                SELECT *, {literal(name)} AS source_file,
                CASE WHEN duration_integer > 0 THEN
                    try(start_wall + cast(duration_integer AS BIGINT) * INTERVAL '1 SECOND')
                END AS end_wall FROM typed"""
        q.run(f"materialize_{table}", f"CREATE TABLE {table} AS {sql}")
        schema = q.run(f"schema_{table}", f"DESCRIBE {table}")
        actual_types = {row["column_name"]: row["column_type"] for row in schema}
        if any(actual_types[name] != "VARCHAR" for name in headers):
            raise ValueError("Raw source columns were not preserved as VARCHAR")
        sources[name] = {
            "logical_role": role, "path": str(path), "bytes": stat.st_size,
            "columns": headers, "raw_column_types": {name: actual_types[name] for name in headers},
            "full_csv_strict_utf8_decode_success": True,
            "manifest_sha256": declared["sha256"], "fresh_csv_sha256_computed": False,
            "manifest_byte_size_matches": True,
            "stat_before": {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        }
        print(f"Read {name}: strict UTF-8, all raw columns VARCHAR", file=sys.stderr, flush=True)
    q.run("epg_union", "CREATE VIEW epg AS SELECT * FROM epg_0 UNION ALL SELECT * FROM epg_1 UNION ALL SELECT * FROM epg_2")
    return sources


def profile_logs(q):
    result = q.one("log_counts_intervals_duration", f"""SELECT count(*) AS rows,
        count(*) FILTER (WHERE {missing('user_id')}) AS missing_user_rows,
        count(*) FILTER (WHERE {missing('vsetv_id')}) AS missing_channel_rows,
        count(DISTINCT user_id) FILTER (WHERE {present('user_id')}) AS distinct_users,
        count(DISTINCT vsetv_id) FILTER (WHERE {present('vsetv_id')}) AS distinct_channels,
        min(start_wall) AS min_start_wall, max(start_wall) AS max_start_wall,
        min(stop_wall) AS min_stop_wall, max(stop_wall) AS max_stop_wall,
        count(*) FILTER (WHERE start_wall IS NULL) AS unparsed_start_rows,
        count(*) FILTER (WHERE stop_wall IS NULL) AS unparsed_stop_rows,
        count(*) FILTER (WHERE stop_wall = start_wall) AS zero_length_interval_rows,
        count(*) FILTER (WHERE stop_wall < start_wall) AS negative_interval_rows,
        count(*) FILTER (WHERE start_wall IS NULL OR stop_wall IS NULL OR stop_wall <= start_wall) AS invalid_interval_rows,
        count(*) FILTER (WHERE {missing('duraton')}) AS missing_duration_rows,
        count(*) FILTER (WHERE {present('duraton')} AND duration_integer IS NULL) AS invalid_or_nonintegral_duration_rows,
        count(*) FILTER (WHERE duration_integer = 0) AS zero_duration_rows,
        count(*) FILTER (WHERE duration_integer < 0) AS negative_duration_rows,
        min(duration_integer) AS min_raw_duration, max(duration_integer) AS max_raw_duration,
        min(date_diff('second', start_wall, stop_wall)) AS min_endpoint_seconds,
        max(date_diff('second', start_wall, stop_wall)) AS max_endpoint_seconds,
        count(*) FILTER (WHERE duration_integer IS NOT NULL AND start_wall IS NOT NULL AND stop_wall IS NOT NULL
            AND duration_integer != date_diff('second', start_wall, stop_wall)) AS duration_endpoint_mismatch_rows,
        count(*) FILTER (WHERE user_id != trim(user_id)) AS outer_whitespace_user_rows
        FROM logs""")
    result["duration_diagnostic_policy"] = "Compare duraton to stop-start in naive wall-clock seconds; do not repair/rescale endpoints"
    result["duration_mismatch_examples"] = q.run("log_mismatch_examples", """SELECT start_time, stop_time, duraton,
        date_diff('second', start_wall, stop_wall) AS endpoint_seconds FROM logs
        WHERE duration_integer != date_diff('second', start_wall, stop_wall)
        ORDER BY start_wall, stop_wall, duraton LIMIT 5""")
    result["weekly_wall_clock"] = q.run("log_weekly_wall_clock", """SELECT strftime(start_wall, '%G-W%V') AS iso_week,
        count(*) AS rows, min(start_wall) AS min_start, max(stop_wall) AS max_stop
        FROM logs WHERE start_wall IS NOT NULL GROUP BY 1 ORDER BY 1""")
    q.run("distinct_log_users", f"CREATE TABLE log_users AS SELECT user_id, count(*) AS log_rows FROM logs WHERE {present('user_id')} GROUP BY user_id")
    q.run("distinct_template_users", f"CREATE TABLE template_users AS SELECT DISTINCT user_id FROM template WHERE {present('user_id')}")
    for table in ("log_users", "template_users"):
        result[table + "_identity"] = q.one(table + "_identity", f"""SELECT count(*) AS distinct_users,
            count(*) FILTER (WHERE regexp_full_match(user_id, '[0-9]+')) AS unsigned_digit_strings,
            count(*) FILTER (WHERE try_cast(user_id AS HUGEINT) > 9223372036854775807
                AND try_cast(user_id AS HUGEINT) <= 18446744073709551615) AS above_signed_int64_within_uint64,
            count(*) FILTER (WHERE try_cast(user_id AS HUGEINT) > 18446744073709551615) AS above_uint64,
            count(*) FILTER (WHERE length(user_id) > 1 AND starts_with(user_id, '0')) AS leading_zero_strings
            FROM {table}""")
    result["template"] = q.one("template_coverage", f"""SELECT (SELECT count(*) FROM template) AS rows,
        (SELECT count(*) FROM template WHERE {missing('user_id')}) AS missing_user_rows,
        (SELECT count(*) FROM template) - (SELECT count(DISTINCT user_id) FROM template) AS duplicate_user_rows,
        count(*) AS distinct_nonmissing_users,
        count(*) FILTER (WHERE l.user_id IS NOT NULL) AS users_present_in_logs,
        count(*) FILTER (WHERE l.user_id IS NULL) AS users_absent_from_logs,
        coalesce(sum(l.log_rows), 0) AS log_rows_for_template_users
        FROM template_users t LEFT JOIN log_users l USING(user_id)""")
    result["template"]["unsigned_user_examples"] = q.run("template_uint64_examples", """SELECT t.user_id, l.log_rows
        FROM template_users t LEFT JOIN log_users l USING(user_id)
        WHERE try_cast(t.user_id AS HUGEINT) > 9223372036854775807 ORDER BY t.user_id LIMIT 5""")
    result["template"]["absent_user_examples"] = q.run("template_absent_examples", """SELECT t.user_id
        FROM template_users t LEFT JOIN log_users l USING(user_id) WHERE l.user_id IS NULL ORDER BY t.user_id LIMIT 5""")
    return result


def epg_counts(q, source):
    return q.one("epg_counts_" + source, f"""SELECT count(*) AS rows,
        count(*) FILTER (WHERE program_id > 0) AS positive_program_rows,
        count(*) FILTER (WHERE program_id = 0) AS zero_program_rows,
        count(*) FILTER (WHERE {missing('tv_show_id')}) AS missing_program_rows,
        count(*) FILTER (WHERE program_id < 0) AS negative_program_rows,
        count(*) FILTER (WHERE {present('tv_show_id')} AND program_id IS NULL) AS invalid_program_rows,
        count(*) FILTER (WHERE program_id > 9223372036854775807) AS program_rows_above_signed_int64,
        count(DISTINCT program_id) FILTER (WHERE program_id > 0) AS distinct_positive_programs,
        count(DISTINCT channel_id) FILTER (WHERE {present('channel_id')}) AS distinct_channels,
        count(DISTINCT channel_id) FILTER (WHERE program_id > 0 AND {present('channel_id')}) AS distinct_positive_program_channels,
        count(*) FILTER (WHERE {missing('channel_id')}) AS missing_channel_rows,
        min(start_wall) AS min_start_wall, max(start_wall) AS max_start_wall, max(end_wall) AS max_end_wall,
        min(start_wall) FILTER (WHERE program_id > 0) AS min_positive_start_wall,
        max(end_wall) FILTER (WHERE program_id > 0) AS max_positive_end_wall,
        count(*) FILTER (WHERE start_wall IS NULL) AS unparsed_start_rows,
        count(*) FILTER (WHERE {missing('duration')}) AS missing_duration_rows,
        count(*) FILTER (WHERE {present('duration')} AND duration_integer IS NULL) AS invalid_or_nonintegral_duration_rows,
        count(*) FILTER (WHERE regexp_full_match(trim(duration), '[+-]?[0-9]+\\.[0-9]+')
            AND NOT regexp_full_match(trim(duration), {literal(INTEGER_TEXT)})) AS fractional_decimal_duration_rows,
        count(*) FILTER (WHERE duration_integer = 0) AS zero_duration_rows,
        count(*) FILTER (WHERE duration_integer < 0) AS negative_duration_rows,
        min(duration_integer) AS min_duration_seconds, max(duration_integer) AS max_duration_seconds,
        count(*) FILTER (WHERE duration_integer > 0 AND start_wall IS NOT NULL AND end_wall IS NULL) AS unrepresentable_end_rows,
        count(*) FILTER (WHERE program_id > 0 AND (start_wall IS NULL OR end_wall IS NULL OR {missing('channel_id')})) AS invalid_positive_airing_rows,
        count(*) FILTER (WHERE cast(start_wall AS DATE) < cast(end_wall AS DATE)) AS crosses_calendar_midnight_rows
        FROM epg WHERE source_file=?""", [source])


def metadata_profile(q):
    fields = {"title": "tv_show_title", "category": "tv_show_category", "genre_1": "tv_show_genre_1",
              "genre_2": "tv_show_genre_2", "genre_3": "tv_show_genre_3", "year": "year_of_production"}
    genre = "(" + " OR ".join(present(f"tv_show_genre_{i}") for i in (1, 2, 3)) + ")"
    aggregations = []
    for short, column in fields.items():
        aggregations.extend([f"bool_or({present(column)}) AS {short}_any", f"bool_and({present(column)}) AS {short}_every"])
    aggregations.extend([f"bool_or({genre}) AS any_genre_any", f"bool_and({genre}) AS any_genre_every"])
    q.run("metadata_by_file_and_id", f"""CREATE TABLE metadata_ids AS SELECT source_file, program_id,
        count(*) AS source_rows, {', '.join(aggregations)} FROM epg WHERE program_id > 0 GROUP BY 1,2""")
    columns = ", ".join(f"count(*) FILTER (WHERE {field}_{policy}) AS programs_with_{field}_{policy}"
                        for field in [*fields, "any_genre"] for policy in ("any", "every"))
    availability = q.run("metadata_availability", f"SELECT source_file, count(*) AS positive_programs, {columns} FROM metadata_ids GROUP BY 1 ORDER BY 1")
    q.run("global_program_identity", f"""CREATE TABLE program_identity AS SELECT program_id,
        count(*) AS source_rows, count(DISTINCT source_file) AS source_files,
        count(DISTINCT channel_id) AS channels,
        count(DISTINCT tv_show_title) FILTER (WHERE {present('tv_show_title')}) AS title_variants,
        count(DISTINCT tv_show_category) FILTER (WHERE {present('tv_show_category')}) AS category_variants,
        min(tv_show_title) FILTER (WHERE {present('tv_show_title')}) AS example_title,
        min(start_wall) AS first_start, max(end_wall) AS last_end
        FROM epg WHERE program_id > 0 GROUP BY 1""")
    identity = q.one("program_identity_summary", """SELECT count(*) AS positive_programs,
        count(*) FILTER (WHERE source_rows > 1) AS repeated_program_ids,
        count(*) FILTER (WHERE title_variants > 1) AS ids_with_multiple_exact_titles,
        count(*) FILTER (WHERE category_variants > 1) AS ids_with_multiple_categories,
        max(title_variants) AS max_title_variants, max(category_variants) AS max_category_variants
        FROM program_identity""")
    examples = q.run("program_identity_top_examples", """SELECT * FROM program_identity
        WHERE title_variants > 1 OR category_variants > 1
        ORDER BY title_variants DESC, category_variants DESC, source_rows DESC, program_id LIMIT 8""")
    for example in examples:
        item = example["program_id"]
        example["title_examples"] = q.run("title_examples_" + str(item), f"""SELECT tv_show_title AS title, count(*) AS source_rows
            FROM epg WHERE program_id=? AND {present('tv_show_title')} GROUP BY 1 ORDER BY 2 DESC,1 LIMIT 5""", [item])
        example["category_examples"] = q.run("category_examples_" + str(item), f"""SELECT tv_show_category AS category, count(*) AS source_rows
            FROM epg WHERE program_id=? AND {present('tv_show_category')} GROUP BY 1 ORDER BY 2 DESC,1 LIMIT 5""", [item])
    identity["examples"] = examples
    identity["variant_definition"] = "Distinct exact nonmissing source strings; no title/category normalization or episode inference"
    identity["year_text"] = q.one("year_text_profile", f"""SELECT count(*) FILTER (WHERE {missing('year_of_production')}) AS positive_rows_missing_year,
        count(*) FILTER (WHERE {present('year_of_production')} AND NOT regexp_full_match(trim(year_of_production), '[0-9]+')) AS positive_rows_nondigit_year_text
        FROM epg WHERE program_id > 0""")
    identity["year_text"]["examples"] = q.run("nondigit_year_examples", f"""SELECT year_of_production AS year_text, count(*) AS rows
        FROM epg WHERE program_id > 0 AND {present('year_of_production')}
        AND NOT regexp_full_match(trim(year_of_production), '[0-9]+') GROUP BY 1 ORDER BY 2 DESC,1 LIMIT 8""")
    return {"availability_per_positive_id": availability, "identity": identity}


def components(q, prefix, *, source=None, include_zero=False, examples=2):
    condition = "program_id >= 0" if include_zero else "program_id > 0"
    parameters = []
    if source is not None:
        condition += " AND source_file=?"
        parameters = [source]
    condition += f" AND {present('channel_id')} AND start_wall IS NOT NULL AND end_wall > start_wall"
    q.run(prefix + "_deduplicate", f"""CREATE TABLE {prefix}_identities AS SELECT channel_id, start_wall, end_wall, program_id,
        count(*) AS source_multiplicity FROM epg WHERE {condition} GROUP BY 1,2,3,4""", parameters)
    duplicates = q.one(prefix + "_duplicates", f"""SELECT count(*) AS unique_airing_identities,
        coalesce(sum(source_multiplicity),0) AS source_rows,
        coalesce(sum(source_multiplicity-1),0) AS duplicate_extra_rows,
        count(*) FILTER (WHERE source_multiplicity > 1) AS duplicated_identity_groups
        FROM {prefix}_identities""")
    # Running maximum handles containment/transitive overlap; lag(end) alone does
    # not. Exact duplicates are removed first, adjacent intervals do not conflict.
    q.run(prefix + "_members", f"""CREATE TABLE {prefix}_members AS WITH preceding AS (
        SELECT *, max(end_wall) OVER (PARTITION BY channel_id ORDER BY start_wall,end_wall,program_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS previous_max_end
        FROM {prefix}_identities), flags AS (
        SELECT *, CASE WHEN previous_max_end IS NULL OR start_wall >= previous_max_end THEN 1 ELSE 0 END AS new_component
        FROM preceding)
        SELECT *, sum(new_component) OVER (PARTITION BY channel_id ORDER BY start_wall,end_wall,program_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS component_id FROM flags""")
    q.run(prefix + "_groups", f"""CREATE TABLE {prefix}_components AS SELECT channel_id, component_id,
        count(*) AS unique_airings, sum(source_multiplicity) AS source_rows,
        count(DISTINCT program_id) AS distinct_ids, min(start_wall) AS first_start, max(end_wall) AS last_end
        FROM {prefix}_members GROUP BY 1,2""")
    overlaps = q.one(prefix + "_overlap_summary", f"""SELECT
        count(*) FILTER (WHERE unique_airings > 1) AS conflict_components,
        coalesce(sum(unique_airings) FILTER (WHERE unique_airings > 1),0) AS conflict_unique_airings,
        coalesce(sum(source_rows) FILTER (WHERE unique_airings > 1),0) AS conflict_source_rows,
        count(*) FILTER (WHERE unique_airings > 1 AND distinct_ids > 1) AS components_with_multiple_program_ids,
        coalesce(max(unique_airings) FILTER (WHERE unique_airings > 1),0) AS largest_component_airings
        FROM {prefix}_components""")
    overlaps.update(q.one(prefix + "_overlap_members", f"""SELECT
        count(*) FILTER (WHERE start_wall < previous_max_end) AS rows_overlapping_an_earlier_interval,
        count(*) FILTER (WHERE end_wall <= previous_max_end) AS rows_contained_by_earlier_envelope
        FROM {prefix}_members"""))
    overlaps["pair_count_computed"] = False
    overlaps["definition"] = "Connected positive-length same-channel overlap components after exact identity deduplication; not a count of pairwise overlaps"
    sample = q.run(prefix + "_component_examples", f"""SELECT * FROM {prefix}_components WHERE unique_airings > 1
        ORDER BY unique_airings DESC,channel_id,first_start LIMIT {int(examples)}""")
    for row in sample:
        row["airing_examples"] = q.run(prefix + "_airing_examples", f"""SELECT m.start_wall, m.end_wall, m.program_id,
            m.source_multiplicity, p.example_title AS example_title_for_program_id FROM {prefix}_members m
            LEFT JOIN program_identity p USING(program_id) WHERE m.channel_id=? AND m.component_id=?
            ORDER BY m.start_wall,m.end_wall,m.program_id LIMIT 6""", [row["channel_id"], row["component_id"]])
    return {"positive_ids_only": not include_zero, "exact_duplicates": duplicates, "overlaps": overlaps, "examples": sample}


def channel_and_catalog(q, sources):
    q.run("channel_envelopes", f"""CREATE TABLE epg_channels AS SELECT channel_id,
        min(start_wall) AS min_start, max(end_wall) AS max_end,
        min(start_wall) FILTER (WHERE program_id > 0) AS min_positive_start,
        max(end_wall) FILTER (WHERE program_id > 0) AS max_positive_end
        FROM epg WHERE {present('channel_id')} GROUP BY 1""")
    result = q.one("log_channel_envelope_coverage", """SELECT count(*) AS log_rows,
        count(*) FILTER (WHERE e.channel_id IS NULL) AS rows_without_any_epg_channel,
        count(DISTINCT l.vsetv_id) FILTER (WHERE e.channel_id IS NULL) AS log_channels_without_any_epg_channel,
        count(*) FILTER (WHERE e.min_positive_start IS NULL) AS rows_without_positive_id_epg_channel,
        count(*) FILTER (WHERE l.stop_wall > l.start_wall AND e.min_start IS NOT NULL
            AND l.start_wall < e.max_end AND l.stop_wall > e.min_start) AS rows_overlapping_any_id_channel_envelope,
        count(*) FILTER (WHERE l.stop_wall > l.start_wall AND e.min_positive_start IS NOT NULL
            AND l.start_wall < e.max_positive_end AND l.stop_wall > e.min_positive_start) AS rows_overlapping_positive_id_channel_envelope,
        count(*) FILTER (WHERE l.stop_wall <= e.min_positive_start) AS rows_ending_before_or_at_first_positive_airing,
        count(*) FILTER (WHERE l.start_wall >= e.max_positive_end) AS rows_starting_after_or_at_last_positive_airing
        FROM logs l LEFT JOIN epg_channels e ON l.vsetv_id=e.channel_id""")
    result["comparison_policy"] = "Exact string channel equality; naive wall-clock envelopes only, not airing joins or proof of shared source time zone"
    result["unmatched_channel_examples"] = q.run("unmatched_log_channel_examples", """SELECT l.vsetv_id, count(*) AS log_rows
        FROM logs l LEFT JOIN epg_channels e ON l.vsetv_id=e.channel_id WHERE e.channel_id IS NULL
        GROUP BY 1 ORDER BY 2 DESC,1 LIMIT 8""")
    result["file_time_envelopes"] = q.run("source_time_envelopes", """WITH ranges AS (
        SELECT source_file,min(start_wall) AS first_start,max(end_wall) AS last_end FROM epg GROUP BY 1)
        SELECT r.source_file,r.first_start,r.last_end,
        (SELECT count(*) FROM logs l WHERE l.start_wall >= r.first_start AND l.start_wall < r.last_end) AS log_starts_inside_file_envelope
        FROM ranges r ORDER BY r.first_start""")
    future = sources[EPG_ROLES[2]]
    result["program_catalog"] = q.one("historical_future_program_overlap", """WITH ids AS (
        SELECT program_id,bool_or(source_file=?) AS in_future,bool_or(source_file!=?) AS in_history
        FROM epg WHERE program_id>0 GROUP BY 1)
        SELECT count(*) AS all_positive_programs,count(*) FILTER (WHERE in_history) AS historical_positive_programs,
        count(*) FILTER (WHERE in_future) AS future_positive_programs,
        count(*) FILTER (WHERE in_future AND in_history) AS future_programs_seen_in_historical_schedules,
        count(*) FILTER (WHERE in_future AND NOT in_history) AS future_only_programs
        FROM ids""", [future, future])
    result["program_catalog"]["history_definition"] = "Presence in either supplied historical EPG export, not observed user history or qualifying viewing"
    return result


def markdown(data):
    logs = data["logs"]
    template = logs["template"]
    global_positive = data["conflicts"]["positive_global"]
    identities = data["metadata"]["identity"]
    coverage = data["channel_and_catalog"]
    text = ["# Independent official-source profile", "", f"Generated: {data['generated_at_local']}. Input identity is attributed to the existing acquisition manifest.", "",
            "## Scope and interpretation", "",
            "Full source scans used DuckDB with four threads, a 4GB engine memory limit, and disposable spill/database files under project `.cache/`. No production parser, viewing reconstruction, model training, authentication, or download was used.", "",
            "**Times below are timezone-unknown source wall clocks, not UTC.** Logs use `start_time`, `stop_time`, and `duraton` (source typo); EPG starts use `%d.%m.%Y %H:%M:%S`, with whole-second duration diagnostics. No timezone offset was inferred or applied.", "",
            "All raw columns were VARCHAR and every CSV completed strict UTF-8 decoding. Users remain exact strings, including values above signed-int64 maximum. Existing manifest SHA-256 values are copied with attribution; CSV bytes were **not rehashed**. Byte sizes and before/after stat stability were checked.", "",
            "## Viewing logs and template", "",
            f"- Logs: **{logs['rows']:,} rows**, **{logs['distinct_users']:,} distinct nonmissing users**, **{logs['distinct_channels']:,} channels**.",
            f"- Wall-clock starts: `{logs['min_start_wall']}` through `{logs['max_start_wall']}`; maximum stop: `{logs['max_stop_wall']}`.",
            f"- Invalid intervals: **{logs['invalid_interval_rows']:,}**; missing users/channels: **{logs['missing_user_rows']:,}/{logs['missing_channel_rows']:,}**.",
            f"- Raw `duraton` negative/zero: **{logs['negative_duration_rows']:,}/{logs['zero_duration_rows']:,}**; endpoint mismatch rows: **{logs['duration_endpoint_mismatch_rows']:,}**; invalid/nonintegral duration rows: **{logs['invalid_or_nonintegral_duration_rows']:,}**.",
            f"- Template: **{template['rows']:,} rows**, **{template['duplicate_user_rows']:,} duplicate IDs**, **{template['users_present_in_logs']:,} distinct users present in logs**, **{template['users_absent_from_logs']:,} absent**.",
            f"- Distinct user IDs above 2^63-1 and within uint64: logs **{logs['log_users_identity']['above_signed_int64_within_uint64']:,}**, template **{logs['template_users_identity']['above_signed_int64_within_uint64']:,}**. Never cast these identifiers to signed int64 or binary floats.", "",
            "## EPG files", "",
            "Positive IDs are exact integral values >0. Missing means blank or an explicitly listed null token. Duplicate/overlap counts below concern valid positive-ID airing intervals, not zero placeholders.", "",
            "| Source | Rows | Positive / zero / missing ID rows | Positive IDs | Channels | First start / last end (wall clock) | Extra positive duplicate rows | Positive conflicted unique airings |",
            "| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |"]
    for name, epg in data["epg"].items():
        conflict = data["conflicts"]["per_file"][name]
        text.append(f"| `{name}` | {epg['rows']:,} | {epg['positive_program_rows']:,} / {epg['zero_program_rows']:,} / {epg['missing_program_rows']:,} | {epg['distinct_positive_programs']:,} | {epg['distinct_channels']:,} | {epg['min_start_wall']} / {epg['max_end_wall']} | {conflict['exact_duplicates']['duplicate_extra_rows']:,} | {conflict['overlaps']['conflict_unique_airings']:,} |")
    text.extend(["", "EPG duration/interval exclusions, in source order:", ""])
    for name, epg in data["epg"].items():
        text.append(f"- `{name}`: invalid/nonintegral duration **{epg['invalid_or_nonintegral_duration_rows']:,}**, zero **{epg['zero_duration_rows']:,}**, negative **{epg['negative_duration_rows']:,}**; unparsed starts **{epg['unparsed_start_rows']:,}**; invalid positive-ID airings **{epg['invalid_positive_airing_rows']:,}**.")
    text.extend(["", "Per-file malformed/nonintegral durations, nonpositive durations, parsed timestamp failures, cross-midnight counts, and source schemas are recorded in the JSON. The actual file ranges, not legacy logical filenames, must drive protocol reconciliation.", "",
                 "## Catalog and metadata", "",
                 f"- Historical EPG program IDs: **{coverage['program_catalog']['historical_positive_programs']:,}**; future IDs: **{coverage['program_catalog']['future_positive_programs']:,}**; future IDs already present in historical schedules: **{coverage['program_catalog']['future_programs_seen_in_historical_schedules']:,}**; future-only IDs: **{coverage['program_catalog']['future_only_programs']:,}**. These are schedule-presence counts, not viewing-derived seen/cold users.", "",
                 "Availability means at least one nonmissing metadata value for a positive ID within that source. JSON also records whether it is present on every row for that ID.", "",
                 "| Source | Positive IDs | Any title | Any category | Any genre slot |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in data["metadata"]["availability_per_positive_id"]:
        text.append(f"| `{row['source_file']}` | {row['positive_programs']:,} | {row['programs_with_title_any']:,} | {row['programs_with_category_any']:,} | {row['programs_with_any_genre_any']:,} |")
    text.extend(["", f"Across all positive IDs, **{identities['ids_with_multiple_exact_titles']:,}** have multiple exact nonmissing titles and **{identities['ids_with_multiple_categories']:,}** multiple categories. Display-title variants do not establish episode-level identity; do not replace raw-ID aggregation with guessed episode identities.", ""])
    for row in identities["examples"][:5]:
        titles = [entry["title"] for entry in row["title_examples"][:2]]
        text.append(f"- ID `{row['program_id']}`: {row['title_variants']} exact title variants, {row['category_variants']} category variants. Examples: {json.dumps(titles, ensure_ascii=False)}.")
    text.extend(["", f"Year metadata is missing on **{identities['year_text']['positive_rows_missing_year']:,}** positive-ID rows and has non-digit text on **{identities['year_text']['positive_rows_nondigit_year_text']:,}** rows (examples include open-ended/range years in JSON). This field is not safely modeled by a blanket integer cast."])
    text.extend(["", "## Duplicates and overlap diagnostics", "",
                 f"Across all EPG files after positive-ID filtering: **{global_positive['exact_duplicates']['duplicate_extra_rows']:,}** extra exact identity rows; **{global_positive['overlaps']['conflict_components']:,}** positive-overlap components containing **{global_positive['overlaps']['conflict_unique_airings']:,}** distinct airings.",
                 "Exact identity is `(channel_id,start,end,program_id)`, excluding metadata. Exact duplicates are removed before the running-maximum interval sweep; adjacency is not overlap. Components include containment and transitive chains. These are **not pairwise overlap counts**, and the profile does not quarantine or relabel anything.",
                 f"Including zero-ID rows would additionally place **{data['conflicts']['zero_inclusion_effect']['additional_positive_airings_in_conflict']:,}** otherwise nonconflicting positive airings into conflict components. Zero placeholders must be excluded before computing positive-program conflict policy. Positive-only conflicts above remain and require a documented decision.", "",
                 "Bounded per-file/global component examples, duplicate multiplicities, dates and raw IDs are in the JSON. Cross-midnight/duplicate-source entries can contribute; observed interval overlap alone does not identify the organizer's intended reconstruction.", "",
                 "## Channel/time availability and policy implications", "",
                 f"- Log rows without any matching raw-string EPG channel: **{coverage['rows_without_any_epg_channel']:,}** across **{coverage['log_channels_without_any_epg_channel']:,}** channels; log rows lacking a positive-ID EPG channel: **{coverage['rows_without_positive_id_epg_channel']:,}**.",
                 f"- Valid log intervals overlapping a positive-ID channel's outer schedule envelope: **{coverage['rows_overlapping_positive_id_channel_envelope']:,}**. This is an envelope check, **not actual airing coverage**, and assumes only a common wall-clock coordinate for the comparison.",
                 "- Use the actual `stop_time`/`duraton` names and day-first EPG parsing. Preserve nonnumeric/range year metadata as text. Reconcile the observed source date ranges before choosing split cutoffs.",
                 "- A shared timezone/UTC offset remains **unknown**. The source has no zone; overlap/envelope statistics cannot justify converting either source to UTC.",
                 "- No production parsing policy was changed. Counts are source diagnostics, not recommendation quality, Kaggle scores, qualifying viewing events, or personal verification.", "",
                 "## Reproduction and evidence", "", "```sh", ".venv/bin/python -B tools/profile_official.py", "```", "",
                 f"Observed total elapsed: **{data['execution']['elapsed_seconds']:.3f} seconds**; DuckDB **{data['execution']['duckdb_version']}**; **{len(data['executed_queries'])}** timed SQL statements. This is one profiling run, not a model/runtime feasibility benchmark.",
                 "`results/official-independent-profile.json` contains the executed SQL/parameters/timings, complete aggregates, exact bounded examples, raw schemas, declared hashes and their verification limits. Output files must be new; reruns require explicitly different output paths.", ""])
    details = ["## Bounded anomaly examples", ""]
    if logs["duration_mismatch_examples"]:
        differences = sorted({entry["endpoint_seconds"] - int(Decimal(entry["duraton"])) for entry in logs["duration_mismatch_examples"]})
        details.extend([f"The **{len(logs['duration_mismatch_examples'])} stored log mismatch examples** have `(stop-start)-duraton` differences of `{differences}` seconds. This does not describe all mismatches or establish a source timezone/DST rule; do not automatically shift or repair timestamps.", ""])
    details.extend(["| Channel | Component wall-clock span | Positive program IDs |", "| --- | --- | --- |"])
    for row in global_positive["examples"]:
        ids = ", ".join(str(airing["program_id"]) for airing in row["airing_examples"])
        details.append(f"| `{row['channel_id']}` | {row['first_start']} / {row['last_end']} | {ids} |")
    if global_positive["examples"] and all(row["first_start"] > logs["max_stop_wall"] for row in global_positive["examples"]):
        details.extend(["", "All stored component examples start after the observed viewing-log end. They are source overlap diagnostics, not reconstructed user events.", ""])
    else:
        details.extend(["", "Component spans and raw IDs are source overlap diagnostics, not reconstructed user events.", ""])
    text.extend(details)
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--file-map", type=Path, default=Path("configs/source-files.json"))
    parser.add_argument("--sample", type=Path, default=Path("results/official-source-sample.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/source-manifest.json"))
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--out", type=Path, default=Path("results/official-independent-profile.json"))
    parser.add_argument("--report", type=Path, default=Path("reviews/official-profile.md"))
    args = parser.parse_args()
    if args.out.exists() or args.report.exists():
        raise FileExistsError("Profile/report outputs must both be new; no overwrite option")
    if args.out.resolve() == args.report.resolve():
        raise ValueError("JSON and Markdown outputs must differ")
    started = perf_counter()
    mapping = json.loads(args.file_map.read_text(encoding="utf-8"))
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    roles = (LOG_ROLE, *EPG_ROLES, TEMPLATE_ROLE)
    if set(mapping) != set(roles) or len(set(mapping.values())) != 5:
        raise ValueError("Expected five distinct explicitly mapped source basenames")
    if any(not isinstance(name, str) or Path(name).name != name or name in (".", "..") for name in mapping.values()):
        raise ValueError("Only explicit source basenames are allowed")
    if mapping != manifest["logical_to_source_filenames"]:
        raise ValueError("File map differs from the acquisition manifest")
    args.cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="official-independent-profile-", dir=args.cache) as temporary:
        connection = duckdb.connect(str(Path(temporary) / "profile.duckdb"))
        q = Queries(connection)
        try:
            q.run("threads", "SET threads=4")
            q.run("memory", "SET memory_limit='4GB'")
            q.run("preserve_order", "SET preserve_insertion_order=true")
            q.run("spill_directory", "SET temp_directory=" + literal(str(Path(temporary) / "spill")))
            settings = q.one("settings", "SELECT current_setting('threads') AS threads, current_setting('memory_limit') AS memory_limit")
            source_profiles = load_sources(q, args.raw_dir, mapping, sample, manifest)
            logs = profile_logs(q)
            epg = {mapping[role]: epg_counts(q, mapping[role]) for role in EPG_ROLES}
            metadata = metadata_profile(q)
            conflicts = {"per_file": {mapping[role]: components(q, f"positive_{i}", source=mapping[role])
                                      for i, role in enumerate(EPG_ROLES)}}
            conflicts["positive_global"] = components(q, "positive_global", examples=5)
            conflicts["including_zero_global"] = components(q, "including_zero", include_zero=True, examples=2)
            conflicts["zero_inclusion_effect"] = q.one("zero_induced_positive_conflicts", """SELECT
                count(*) FILTER (WHERE ac.unique_airings > 1 AND pc.unique_airings = 1) AS additional_positive_airings_in_conflict,
                count(*) FILTER (WHERE ac.unique_airings > 1) AS positive_airings_in_conflict_when_zero_included
                FROM positive_global_members p
                JOIN positive_global_components pc USING(channel_id,component_id)
                JOIN including_zero_members a USING(channel_id,start_wall,end_wall,program_id)
                JOIN including_zero_components ac ON a.channel_id=ac.channel_id AND a.component_id=ac.component_id""")
            coverage = channel_and_catalog(q, mapping)
            for name, profile in source_profiles.items():
                stat = (args.raw_dir / name).stat()
                profile["stat_after"] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                profile["source_stat_unchanged"] = profile["stat_before"] == profile["stat_after"]
                if not profile["source_stat_unchanged"]:
                    raise RuntimeError(f"Source changed during profiling: {name}")
        finally:
            connection.close()
    data = {
        "generated_at_local": datetime.now().astimezone().isoformat(),
        "observed_source_profile": True, "production_parser_imported": False,
        "viewing_reconstruction_performed": False, "model_training_performed": False,
        "timestamp_policy": "Naive source wall-clock TIMESTAMP; no timezone conversion, UTC assertion, or inferred offset",
        "missing_value_policy": "Blank or case-insensitive trimmed null/none/nan/na/n/a/backslash-N tokens; exact other user/channel strings preserved",
        "identifier_policy": "All raw CSV columns VARCHAR; HUGEINT used only for numeric diagnostics/program grouping, never user identity or joins",
        "hash_policy": "Existing acquisition-manifest CSV SHA-256 values attributed, not recomputed; current sizes and before/after stat checked",
        "source_claim_basis": {"competition": manifest.get("competition"), "acquisition": manifest.get("acquisition"),
                               "independent_remote_authentication_performed": False},
        "input_metadata_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (args.file_map, args.sample, args.manifest)},
        "logical_to_source_filenames": mapping, "files": source_profiles,
        "logs": logs, "epg": epg, "metadata": metadata, "conflicts": conflicts,
        "channel_and_catalog": coverage,
        "execution": {"duckdb_version": duckdb.__version__, "python_version": platform.python_version(),
                      "settings": settings, "cache_parent": str(args.cache), "temporary_database_and_spill_removed": True,
                      "elapsed_seconds": perf_counter() - started,
                      "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if sys.platform == "darwin" else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      "memory_limit_scope": "DuckDB engine buffer limit, not a claim of a strict process-RSS ceiling"},
        "executed_queries": q.records,
        "limitations": ["No source timezone is supplied or inferred", "Channel envelopes are not exact airing coverage",
                        "No pairwise overlap count or user-airing reconstruction", "Exact title variants do not determine episode granularity",
                        "Manifest CSV hashes were not independently rehashed", "No recommendation quality, Kaggle score, or personal-verification claim"],
    }
    # Convert wall-clock datetime/Decimal objects before rendering or serializing.
    data = json.loads(json.dumps(data, default=json_default, ensure_ascii=False, allow_nan=False))
    report = markdown(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with args.report.open("x", encoding="utf-8") as stream:
        stream.write(report)
    print(json.dumps({"profile": str(args.out), "report": str(args.report), "log_rows": logs["rows"],
                      "positive_conflict_airings": conflicts["positive_global"]["overlaps"]["conflict_unique_airings"],
                      "elapsed_seconds": data["execution"]["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
