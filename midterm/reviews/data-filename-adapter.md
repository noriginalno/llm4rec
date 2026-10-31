# Explicit source-filename adapter

Date: 2026-10-31. The optional filename adapter is implemented; the library's
five logical filename defaults and the data/reconstruction interfaces remain
backward-compatible. No source files are renamed, copied, rewritten or discovered
heuristically by `load_raw`.

## Newly verified metadata versus still-unknown contents

The parent session reported that authenticated official CLI command
`competitions files sweettv-tv-program-recommender` listed the following actual
filenames and byte counts after browser OAuth succeeded. These are user-carried
official-listing facts, not a CLI run or authentication performed by this author.

| Existing logical role/default | Actual listed basename | Listed bytes |
| --- | --- | ---: |
| `dataset11-30week.csv` | `dataset11-30.csv` | 396216555 |
| `export_arh_10-19.csv` | `export_arh_11-20-final.csv` | 153322643 |
| `export_arh_20-30.csv` | `export_arh_21-30-final.csv` | 154366907 |
| `export_arh_31-42.csv` | `export_arh_31-42-final.csv` | 185952991 |
| `submission.csv` | `submission.csv` | 38348 |

The handoff reports that downloads still return 403 pending the user's rule
acceptance. **No actual CSV contents, header bytes/version, timestamp units or
time zone have been inspected.** This listing does not establish date/week
semantics, row counts, encodings, schemas or organizer reconstruction policies.
The documented column-name defaults stay unchanged, but they are not promoted
to actual-source verification: `schema_verified_on_actual_source` remains false.
No actual-source SHA-256 values are known from the listing alone.

## API and configuration

`load_raw(raw_dir, ..., file_names: Mapping[str, str] | None = None)` accepts
exact mappings from the **values** of `LOG_FILE`, `SCHEDULE_FILES` and
`TEMPLATE_FILE` to actual relative basenames. Keys are not Python variable names.
For the listing above, a caller can explicitly provide:

```python
source_files = {
    "dataset11-30week.csv": "dataset11-30.csv",
    "export_arh_10-19.csv": "export_arh_11-20-final.csv",
    "export_arh_20-30.csv": "export_arh_21-30-final.csv",
    "export_arh_31-42.csv": "export_arh_31-42-final.csv",
    "submission.csv": "submission.csv",
}
prepared = load_raw(raw_dir, file_names=source_files)
```

This example expresses the requested role mapping explicitly; it does not infer
temporal cutoffs or validate that similarly numbered export names have the same
date ranges. No parent CLI/configuration/download files were changed here.

- `None`, `{}`, and omitted logical keys preserve the corresponding original
  logical default basenames. Caller-provided mappings are copied, not mutated.
- Validation occurs before file access. Non-mappings, unknown logical keys,
  non-string/blank values, `.`/`..`, slash/backslash paths, Windows drive prefixes
  and NUL characters raise `DataSchemaError` (a `ValueError` subclass).
- The five effective basenames must be distinct. Collisions with omitted default
  roles are also rejected. Comparison is exact; no case folding, filename
  normalization, suffix matching or directory scanning is used.
- Missing-file errors list the effective **actual basenames**. An explicit
  alias never falls back to an existing logical-name file. Parsing errors also
  identify the actual source basename.
- `encoding`, `delimiter` and `timestamp_unit` mappings remain keyed by exact
  **logical names**, even when source files are aliased. The actual aliases are
  not parsing-option keys. Global options retain their existing behavior.
- Filename mapping does not change `RawSchema`, timestamp parsing, event union,
  conflict quarantine or fold policies. No units/time zones are inferred from a
  filename or file size.

## Audit and hashes

`prepared.audit["file_names"]` records all five effective logical-to-actual
bindings, including identity/defaulted ones. `audit["files"]` remains keyed by
logical name for compatibility. Each file record includes:

```python
prepared.audit["files"][LOG_FILE]["logical_filename"]  # "dataset11-30week.csv"
prepared.audit["files"][LOG_FILE]["source_filename"]   # "dataset11-30.csv"
```

The existing `sha256` and `size_bytes` fields are calculated from the **actual
opened file's bytes**, not the logical name, listing metadata or a substitute
file. CSV row/header/format audits describe that same source. A partial parsing
error carries the effective mapping and actual filename through
`DataSchemaError.audit`. Missing-file checks precede parsing and raise the
existing plain `FileNotFoundError`.

## Executed checks

Using synthetic contents with literal documented `start_time`/`end_time` column
names, the added author cases cover the listed actual filenames, byte-exact
hashes and source preservation, template aliases, partial/empty maps, logical
option dispatch, unknown keys, unsafe names, duplicate effective basenames,
actual-name missing/schema errors, and rejection of heuristic fallback. The
logical-option dispatch fixture requires explicit `^` delimiters, epoch-0
seconds and CP1251 Cyrillic metadata, so incorrect alias-key dispatch cannot
pass accidentally via defaults.

From `midterm/`:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_data.py --tb=short -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_data_adversarial.py --tb=short -p no:cacheprovider
```

- Final author suite: **74 passed in 0.21s** (45 existing cases plus 29 filename
  adapter cases).
- Unchanged adversarial suite: **41 passed in 0.19s** against the same production
  implementation. Only an additional author no-fallback case was added afterward;
  production and independent tests were unchanged.
- Adversarial file SHA-256 was asserted unchanged:
  `d6726efbe399fa74d428b9de890a6741ed1a3b88d10d33c1ade63daddaa2252a`.
- Production `sweettv/data.py` SHA-256:
  `9da28e877a38e977c432c4f549e5e8f53981839698c55ac0f611612d758c15f8`.
- Author `tests/test_data.py` SHA-256:
  `45bf71293727ef1dcf75f647ea280d2229edaa5205e5dbc2e2dfe58212a79616`.

The filename-validation work is constant-sized for five roles, aside from string
lengths. Reconstruction complexity and its in-memory O(L+S+E) limit are unchanged;
listed byte counts alone do not establish runtime or peak-memory feasibility.
These are author-executed synthetic checks, not real-data or new independent
acceptance evidence. No authentication, downloads, installation, delegation,
submission or publication occurred in this implementation stage.
