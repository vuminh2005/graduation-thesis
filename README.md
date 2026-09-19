# MLTool

MLTool validates and prepares tabular supervised-ML projects, materializes
reproducible feature sets, and compares isolated AutoGluon model families on a
shared validation split. It keeps the test split untouched for a later phase.

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
/path/to/graduation-thesis/.venv/bin/mltool prepare
/path/to/graduation-thesis/.venv/bin/mltool features
/path/to/graduation-thesis/.venv/bin/mltool plan
/path/to/graduation-thesis/.venv/bin/mltool train
/path/to/graduation-thesis/.venv/bin/mltool leaderboard
/path/to/graduation-thesis/.venv/bin/mltool tune
/path/to/graduation-thesis/.venv/bin/mltool tuning-leaderboard
/path/to/graduation-thesis/.venv/bin/mltool finalize
/path/to/graduation-thesis/.venv/bin/mltool final-result
/path/to/graduation-thesis/.venv/bin/mltool register
/path/to/graduation-thesis/.venv/bin/mltool status
/path/to/graduation-thesis/.venv/bin/mltool logs
/path/to/graduation-thesis/.venv/bin/mltool best
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

`prepare` always stops on invalid data, regardless of `validation.enabled` or
the currently reserved `validation.fail_on_error` setting. With
`split.stratify: auto`, binary and multiclass targets are stratified while
regression targets are not. An enabled external preprocessor is loaded from a
config-relative `<python-file>:<class-name>` entrypoint, fitted once on train
features only, and reused for all three transformations.

Prepared artifacts are written to `.mltool/prepared/`. Regression targets that
contain numeric-looking strings remain unchanged during preparation. TODO for
the later training phase: decide whether its model integration requires an
explicit numeric target representation.

`features` reads `.mltool/prepared/` without splitting or preprocessing again.
Each configured feature plugin is instantiated separately per FeatureSet,
fitted on that set's train base features only, and independently transforms the
same base columns for train, validation, and test. Plugin outputs are never
chained: plugin order controls only generated-column concatenation order.
FeatureSet artifacts and lineage manifests are written atomically under
`.mltool/features/`.

`plan` validates that the materialized FeatureSets still match the current
dataset and feature configuration, then prints the deterministic FeatureSet ×
model Cartesian product without creating training output. `train` executes
those candidates sequentially with AutoGluon Tabular on CPU. Every candidate
receives only its FeatureSet train split during `fit`; MLTool predicts and
computes configured metrics on the common validation split afterward. HPO,
bagging, stacking, and AutoGluon's weighted ensemble are disabled so each
candidate represents exactly one configured family.

Training artifacts live under `.mltool/training/`, including one predictor and
result per candidate plus JSON/CSV global leaderboards. `leaderboard` is
read-only. Phases 4 and 5 never load or evaluate FeatureSet `test.parquet` files.

`tune` (Phase 5) needs an optional `hpo:` section, which `init` does not write:

```yaml
hpo:
  top_n: 3               # best Phase-4 SUCCEEDED candidates to tune (default 3)
  num_trials: 10         # HPO trials per candidate (default 10)
  time_limit_seconds: 600  # required; hard budget per candidate
```

It refuses to run on stale training artifacts, re-fits each selected candidate
on its FeatureSet train split with a local, sequential random search
(`num_trials`, `time_limit_seconds`), and evaluates on the validation split only.
Bagging, stacking, weighted ensembles, GPUs, and test data stay off. Params you
fix in a model's `params` stay fixed; the rest of AutoGluon's default search
space is tuned. Families without a default AutoGluon search space (RF, XT) train once with
default hyperparameters; `tune` warns about each such candidate and marks it
`hpo_effective: false` (plus the `hpo_warning` text) in `result.json` and
`selected.json`.

Output goes to `.mltool/tuning/` (`candidates/`, `leaderboard.json/csv`,
`manifest.json`, and `selected.json`, the best tuned configuration for a later
final refit). `tuning-leaderboard` is read-only.

Two different holdouts are involved. Inside each candidate, AutoGluon picks the
winning trial on its own internal holdout carved from the train split.
Cross-candidate ranking (`top_n` selection and the tuning leaderboard) uses
MLTool's validation split. A tuned score can therefore come out lower than the
same candidate's untuned Phase 4 score, and `selected.json` is the best tuned
configuration, not a guarantee of improvement over Phase 4.

Set `training.seed` (default `null`) to control randomness in both `train` and
`tune`. AutoGluon 1.6 has no seed argument on `fit`, so MLTool passes it as the
learner's `random_state` (internal holdout split) and as the model's own seed
hyperparameter (`seed`, `random_state` or `random_seed` depending on family);
a seed you fix in a model's `params` wins. When unset, AutoGluon's own default
seed of 0 applies. Training and tuning artifacts record `seed` (`training.seed`)
and `effective_seed` (the seed the model's hyperparameters actually received,
after any per-model override, or `null` when MLTool passed none; per candidate in `result.json`, keyed by candidate
id in `manifest.json`).

`finalize` (Phase 6) is the only command that evaluates on the test split. It
reads `.mltool/tuning/selected.json` and refuses to run if the tuning, training,
feature or prepared artifacts are missing or stale relative to the current
config. It then:

1. re-runs the Phase 2 split on the raw dataset (same seed and stratification)
   and checks the row counts against the prepared manifest (train+validation
   and test), failing if they differ;
2. combines the raw train and validation rows, refits the external preprocessor
   on them only, and transforms both the combined rows and the test rows;
3. refits only the selected FeatureSet's plugins on the combined rows with the
   same instantiate/fit/transform contract as `features`;
4. runs one AutoGluon fit of the selected family with `best_hyperparameters`
   fixed (no HPO, no bagging/stacking/weighted ensemble, no GPU) and the
   selected candidate's `effective_seed`;
5. predicts on the test rows once and computes the configured metrics.

A non-bagged AutoGluon fit keeps an internal holdout out of training, so
`finalize` passes `refit_full`: the same model is retrained with the same
hyperparameters on every combined row and becomes the predictor's best model
(`LightGBM_FULL`, for example). It reads the raw dataset, never the persisted
`.mltool/features/*/test.parquet`. Output goes to `.mltool/final/`
(`predictor/`, `result.json`, `manifest.json`); its manifest is the only one
that records `test_data_used: true`. `final-result` is read-only and warns when
the config or the tuning selection has changed since `finalize`. The refit
preprocessor and plugin states are not persisted, so the predictor alone cannot
score raw data yet.

`finalize` refuses to run when `.mltool/final/manifest.json` already exists,
because every run evaluates the test split again. Run `mltool register` first
to keep the current result, then pass `--force` to refit and evaluate again.

Phase 7 adds tracking around the commands above; it never changes what they
compute.

- **Execution state** (`.mltool/state.db`, SQLite): one `runs` row per
  `prepare`, `features`, `plan`, `train`, `tune`, `finalize` (including refused
  ones, status `BLOCKED`) and `register`, with `started_at`, `finished_at`,
  `status` (`SUCCEEDED`/`FAILED`/`BLOCKED`), `exit_code` and JSON `details`.
  A command that created nothing (for example a first `validate`, or a failed
  first `prepare`) does not create `.mltool/` just to log itself.
- **MLflow** (`.mltool/mlflow`, local `file:` store; the project name is the
  experiment): `train` logs one run per candidate, `tune` one per tuned
  candidate (best hyperparameters as `hp.*` params), and a successful
  `finalize` exactly one run (`test_*` metrics, `test_data_used=true`). Tracking
  is best-effort: a failure prints a warning and never aborts the command.
  MLflow's own Model Registry is not used, and MLflow >= 3.7 requires
  `MLFLOW_ALLOW_FILE_STORE=true` for a file store, which MLTool sets for you.
- **Registry** (`.mltool/registry/<n>/`): `register` copies `.mltool/final/`
  (predictor, `preprocessor.pkl`, `feature_plugins/`, `result.json`,
  `manifest.json`) into the next integer version and writes `metadata.json`
  (timestamp, git commit when available, config fingerprint, selected
  candidate, best hyperparameters, seeds, test metrics). Versions only grow, so
  registering before each `finalize --force` keeps the history.
- `status` shows per-phase artifact freshness and the last run of each command,
  `logs [--limit N]` the run history (newest first), and `best` the finalized
  model plus the latest registered version. None of them write anything.

Changing the raw dataset invalidates prepared input and requires another
`mltool prepare`. Phase 2 does not currently fingerprint external preprocessor
source code, so changing that code also requires the user to rerun preparation.

The validation command exits with `0` for a valid project, `2` for expected
configuration or dataset errors, and `1` for an unexpected runtime failure.

MLTool currently includes Phase 1 validation, Phase 2 preparation, Phase 3
feature materialization, and Phase 4 isolated candidate training/leaderboards.
Phase 5 adds HPO on the top candidates and selection of one configuration, Phase 6 refits that configuration on train+validation and evaluates it on the
test split, and Phase 7 adds SQLite run state, MLflow tracking and a local model
registry.
