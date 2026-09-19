# MLTool

MLTool Phase 1 is a command-line validator for tabular supervised-ML projects.
It loads a small YAML configuration, reads CSV or Parquet data, computes a
SHA-256 file fingerprint, and reports schema, target, errors, and warnings.

## Install for development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

## Use

```bash
mkdir demo && cd demo
/path/to/graduation-thesis/.venv/bin/mltool init
# Put the configured CSV or Parquet file in place and edit mltool.yaml as needed.
/path/to/graduation-thesis/.venv/bin/mltool validate
```

Run `mltool validate` from the directory containing `mltool.yaml`; Phase 1 does
not perform parent-directory config discovery. Dataset paths inside the config
are resolved relative to that config file. With `format: auto`, only `.csv` and
`.parquet` extensions are recognized. An explicit format selects its parser
regardless of the filename extension.

CSV uses pandas' standard comma-delimited parsing. Regression targets with
numeric-looking strings are accepted when every non-null value can be converted
to a number. Classification imbalance is warned when the smallest class is less
than 10% of the non-null target values.

The validation command exits with `0` for a valid project, `2` for expected
configuration or dataset errors, and `1` for an unexpected runtime failure.

Only project initialization and data validation are included in Phase 1.
