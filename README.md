# MLTool

MLTool validates and prepares tabular supervised-ML projects, materializes
reproducible feature sets, and compares isolated AutoGluon model families on a
shared validation split. It tunes the best candidates and selects one
configuration, then `finalize` refits it on train+validation and evaluates it
once on the test split; every earlier phase leaves that split untouched.

## Install for development

Requires **Python >= 3.12** (matches `pyproject.toml`). The install is large,
about 1.3 GB on disk (mostly AutoGluon's CatBoost/SciPy dependencies and MLflow),
and MLTool has only been tested on Linux x86_64.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

## Use

```bash
mkdir demo && cd demo
/path/to/graduation-thesis/.venv/bin/mltool init
# Put the configured CSV or Parquet file in place (see "Input data" below) and
# edit mltool.yaml as needed.
/path/to/graduation-thesis/.venv/bin/mltool validate
/path/to/graduation-thesis/.venv/bin/mltool prepare
/path/to/graduation-thesis/.venv/bin/mltool features
/path/to/graduation-thesis/.venv/bin/mltool plan
/path/to/graduation-thesis/.venv/bin/mltool train
/path/to/graduation-thesis/.venv/bin/mltool leaderboard
# `tune` needs an `hpo:` section, which `init` does not write. Append this to
# mltool.yaml first (a per-candidate time budget in seconds is required):
#   hpo:
#     time_limit_seconds: 60
/path/to/graduation-thesis/.venv/bin/mltool tune
/path/to/graduation-thesis/.venv/bin/mltool tuning-leaderboard
/path/to/graduation-thesis/.venv/bin/mltool finalize
/path/to/graduation-thesis/.venv/bin/mltool final-result
/path/to/graduation-thesis/.venv/bin/mltool register
/path/to/graduation-thesis/.venv/bin/mltool status
/path/to/graduation-thesis/.venv/bin/mltool logs
/path/to/graduation-thesis/.venv/bin/mltool best
```

**Input data.** Provide a CSV or Parquet file with one column per feature plus
exactly one target column whose name matches `task.target` (the `init` template
assumes `./data/dataset.csv` with a target column named `label`). At least one
non-target column is required. Unknown or misspelled keys in `mltool.yaml` are
rejected with `unsupported "<section>" setting(s): <key>`.

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
contain numeric-looking strings remain unchanged during preparation; `train`,
`tune` and `finalize` convert them strictly to numbers on their own copies (the
result files record `target_conversion_applied`).

`features` reads `.mltool/prepared/` without splitting or preprocessing again.
Each configured feature plugin is instantiated separately per FeatureSet,
fitted on that set's train base features only, and independently transforms the
same base columns for train, validation, and test. Plugin outputs are never
chained: plugin order controls only generated-column concatenation order.
FeatureSet artifacts and lineage manifests are written atomically under
`.mltool/features/`.

`plan` validates that the materialized FeatureSets still match the current
dataset, the prepared artifacts they were built from, and the feature
configuration, then prints the deterministic FeatureSet ×
model Cartesian product without creating training output. `train` executes
those candidates sequentially with AutoGluon Tabular on CPU. Every candidate
receives only its FeatureSet train split during `fit`; MLTool predicts and
computes configured metrics on the common validation split afterward. HPO,
bagging, stacking, and AutoGluon's weighted ensemble are disabled so each
candidate represents exactly one configured family.

One `models[]` entry may use `family: ENSEMBLE` instead of a single family. It
is still one candidate scored on the same validation split, competing in the
same leaderboards and `top_n` selection as every other candidate; it is not a
mode of `finalize`, and enabling ensembling only there would let a model reach
the test set without ever being compared on validation, which MLTool does not
allow. An ENSEMBLE candidate lets AutoGluon itself bag, stack, and
weighted-ensemble several of the five families on the FeatureSet's train split
(`num_gpus=0`, sequential fold fitting, `dynamic_stacking` off, and no
`tuning_data`, exactly like every other candidate):

```yaml
models:
  - name: ag_ensemble
    family: ENSEMBLE
    params:
      families: [GBM, CAT, XGB, RF, XT]   # default: all five, this order; a
                                          # non-empty unique subset (ENSEMBLE
                                          # itself may not appear)
      num_bag_folds: 3                    # default 3; 0 (no bagging) or >= 2
      num_stack_levels: 1                 # default 1; >= 0; > 0 requires
                                          # num_bag_folds >= 2
```

Each member family uses AutoGluon's own defaults plus `training.seed` on its
own seed key; fixed per-family hyperparameters inside an ensemble are not
supported. An ENSEMBLE candidate costs noticeably more time and memory than a
single family, since it fits every member family (repeated across bag folds
and stack levels) instead of one model — on the ~900-row real-data smoke test
used to verify this it took roughly 6-10x as long as one single-family
candidate, still under 9 GB of the machine's ~14 GB RAM. HPO is not applied to
it: combining bagging/stacking with AutoGluon's hyperparameter search is out of
scope, so if an ENSEMBLE candidate reaches `tune`'s `top_n`, it is refit with
its configured ensemble settings and marked `hpo_effective: false`, the same
`result.json`/`selected.json` convention RF and XT already use for "no search
space". If it wins, `finalize` refits the whole ensemble on train+validation
and, exactly like a single family, uses `refit_full` to collapse it into one
`_FULL` predictor trained on every row before the one test evaluation; the
registry, `final-result`/`best`, and the MLflow run all show `family: ENSEMBLE`
and the ensemble config where a single family would show its hyperparameters.

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

**Cross-validated evaluation.** A single validation holdout is small: on the
891-row Titanic set it is 134 rows, and differences between FeatureSets on it
are mostly noise. Adding an optional `evaluation.cv` section scores every
candidate on folds of the development set (the train + validation rows) instead,
and ranks the leaderboards by the mean:

```yaml
evaluation:
  primary_metric: roc_auc
  cv:
    folds: 5      # integer >= 2
    repeats: 1    # integer >= 1; each repeat re-partitions with its own
                  # seed derived from split.random_seed
```

Absent, everything behaves exactly as before. With it:

- Folds are stratified for classification and plain K-fold for regression; the
  fold assignment is deterministic for a given `split.random_seed` and its
  fingerprint is recorded in the manifest.
- **Every fold refits everything.** The external preprocessor and the
  FeatureSet's plugins are fitted on that fold's training rows only and then
  applied to the held-out rows, through the same helpers `finalize` uses. The
  materialized `.mltool/features/` artifacts are deliberately not used for
  scoring, because their plugins saw the whole train split.
- The test rows are never part of the development set, so `train` and `tune`
  still record `test_data_used: false`.
- `train` ranks by the CV mean and shows `mean +/- std (n folds)`; per-fold
  metrics are persisted in each candidate's `result.json`. Fold predictors are
  scratch and are not persisted (nothing downstream loads a training predictor),
  so a cross-validated `result.json` has `predictor_path: null` and
  `predictor_persisted: false`.
- `tune` selects `top_n` by CV mean, runs HPO exactly as before (an AutoGluon
  search on the train split), then re-scores the winning configuration across
  the same folds with its hyperparameters **fixed**; that CV mean is what the
  tuned leaderboard and `selected.json` report. Families HPO does not tune
  (RF, XT, ENSEMBLE) are cross-validated as configured. Note the mild optimism
  this leaves: the hyperparameters were searched on rows that also appear in the
  CV training folds, so a tuned CV mean is not a fully unbiased estimate. Nested
  cross-validation is out of scope.
- `finalize` is unchanged: it refits the selected configuration on
  train+validation and evaluates the test split once.
- Changing `evaluation.cv` is part of the config signature, so it makes training
  and tuning artifacts stale.
- MLflow records the CV mean, a `<metric>_std`, and each fold's value as a
  `fold_<metric>` series stepped by fold index.
- Folds run sequentially to bound memory. `train` and `tune` print the total
  number of fits up front and one progress line per fold, both on stderr.

**Replacing a column with a plugin output.** A feature plugin only ever adds
columns, so a plugin that reads `Age` to build `Age_group` would normally force
`Age` to stay in the FeatureSet. `plugin_inputs` lists columns the plugins may
read that are not features themselves:

```yaml
features:
  sets:
    - name: replaced
      source_columns: [Pclass, Sex, Fare, Embarked]
      plugin_inputs: [Age, SibSp, Parch]
      plugins: [age_group, family_count]
```

The plugins see `source_columns` plus `plugin_inputs`; the FeatureSet's final
columns are `source_columns` plus the plugin outputs, so here `Age_group` and
`family_count` genuinely replace `Age`, `SibSp` and `Parch`. The columns must
exist after preprocessing, must not be the target, must be unique, and a plugin
may not emit a name that collides with a source column or a plugin input.
`plugin_inputs` is part of the FeatureSet recipe, so changing it makes the
feature artifacts stale, and it is honored identically by `features`,
`finalize`'s refit and each cross-validation fold.

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

Feature artifacts are fingerprinted against `.mltool/prepared/`'s parquet files,
not just against the raw dataset, so re-running `prepare` alone (a changed
`split.random_seed` or ratio leaves the raw file untouched) marks them stale and
`plan`, `train`, `tune` and `finalize` all refuse until `mltool features` is run
again. `.mltool/prepared/` must therefore stay in place for those commands.

**Migration note:** a project whose `.mltool/features/manifest.json` was written
before this fingerprint existed reports "feature artifacts are stale ... predates
prepared-artifact fingerprinting". Run `mltool features` once to re-materialize;
nothing else needs changing.

`finalize` records the fingerprints it relied on, so `final-result`, `best` and
`status` report the final model as stale once the dataset, the prepared
artifacts or the selected FeatureSet change underneath it, instead of printing a
superseded test metric as current. `register` refuses a stale final model unless
given `--force`, and a forced registration records the warning it overrode in
`metadata.json` (never `null`) together with `forced: true`.

`final-result` and `best` state whether the selected family was really tuned:
families with no AutoGluon search space (RF, XT) print
`Tuned: no (family RF has no HPO search space)`, and `hpo_effective` /
`hpo_warning` are carried from `selected.json` into `.mltool/final/result.json`,
its manifest, the registry `metadata.json` and the final MLflow run's tags.
`tune` and `finalize` also write `mlflow.json` next to their artifacts, mapping
each candidate to the MLflow run that recorded it.

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
