# Exploring ZTF transients to find Type Ia supernovae

A five-part, hands-on learning project: exploring the **ZTF Bright Transient Survey (BTS)** catalogue with **pandas**, **matplotlib** and **seaborn**, finding the patterns that set Type Ia supernovae apart, building a first, honestly evaluated machine-learning classifier with **scikit-learn**, comparing a dozen models including a neural network, and putting the classifier to work under a daily follow-up capacity limit.

## The notebooks

| # | Notebook | What it teaches | Main finding |
|---|---|---|---|
| 1 | [01_loading_and_cleaning.ipynb](01_loading_and_cleaning.ipynb) | pandas: `read_csv`, missing values, `.str` / `.dt`, masks, `groupby`, `crosstab`, astropy coordinates, Parquet | 33 rows have corrupted `b`/`A_V`. 13 duplicate objects. 2026 types missing from the export. The IAU name leaks the label. |
| 2 | [02_visualization.ipynb](02_visualization.ipynb) | matplotlib (Figure/Axes, projections), seaborn (`histplot`, `kdeplot`, `boxplot`, `heatmap`, `pairplot`), plotly | Ia have narrow timescales and uniform peak brightness (standard candles). The catalogue is magnitude-limited, which creates Malmquist bias. |
| 3 | [03_patterns_and_classifier.ipynb](03_patterns_and_classifier.ipynb) | scikit-learn: baselines, pipelines, cross-validation, metrics, thresholds, permutation importance, leakage, time-based validation | ROC AUC ≈ 0.90 from light-curve timescales alone (≈ 0.85 on future years). Redshift features leak. Data drift in 2022. |
| 4 | [04_experimenting_with_models.ipynb](04_experimenting_with_models.ipynb) | 15 classifiers (including XGBoost and LightGBM) compared with `cross_validate`, leakage per model, `GridSearchCV`, neural networks (`MLPClassifier`, training curves, seeds, ensembles), bootstrap intervals | Every flexible model hits the same ceiling (CV AUC ≈ 0.89). Leakage inflates every model, and missingness itself leaks. The neural network comes within 0.01 of gradient boosting but doesn't beat it. |
| 5 | [05_working_under_capacity.ipynb](05_working_under_capacity.ipynb) | Decision rules on top of a model (top-k, thresholds, expected-false-positive budgets), day-by-day simulation, calibration (`CalibratedClassifierCV`), class-mix (prior) shift, monitoring | "100 spectra a day, ≤ 10 false positives" is met with room to spare at this dataset's 68% Ia, but only above about 65% Ia on recent data. Below that, a good rule sends fewer. |

**[pipeline/](pipeline/README.md)** turns notebooks 1–5 into production-style code: discoveries are replayed through Kafka in simulated real time, a classifier service picks at most 100 candidates a night at ≥ 95% precision, and an evaluator cross-checks every decision with `bts.csv`. On the 2024–25 stream it reaches 96.2% precision, and it survives a mid-night crash without losing candidates.

Run the notebooks in order: notebook 1 writes `data/bts_clean.parquet`, which notebooks 2–5 read.

Each notebook ends with a **"What I learned"** summary and **exercises**.

## Setup

A virtual environment with every library is already in `.venv/`. To recreate it from scratch:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

To open the notebooks, use either:

* **VS Code**: open a `.ipynb` file and choose the `.venv` Python interpreter as the kernel (top right), or
* **JupyterLab**: `.venv\Scripts\jupyter lab`

To re-run every notebook from top to bottom and refresh the HTML reports:

```powershell
.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace 01_loading_and_cleaning.ipynb 02_visualization.ipynb 03_patterns_and_classifier.ipynb 04_experimenting_with_models.ipynb 05_working_under_capacity.ipynb
.venv\Scripts\jupyter nbconvert --to html --output-dir reports 01_loading_and_cleaning.ipynb 02_visualization.ipynb 03_patterns_and_classifier.ipynb 04_experimenting_with_models.ipynb 05_working_under_capacity.ipynb
```

## Project layout

```
bts.csv                          raw data (read-only, never modified)
01_… to 05_….ipynb              the learning notebooks (executed, with outputs)
ztf_style.py                     shared colours and chart style for every figure
data/bts_clean.parquet           cleaned table written by notebook 1
figures/*.png                    every figure, saved at 200 dpi for slides
reports/*.html                   notebooks exported as standalone web pages
requirements.txt                 exact library versions used
```

## Presenting this work

* **`reports/*.html`** open in any browser, with no Python needed. Share these for the full story with code.
* **`figures/*.png`** are ready to drop into slides. Every chart title states its finding.

A suggested 13-slide storyline, with the figure for each slide:

| Slide | Message | Figure |
|---|---|---|
| 1 | The data: 11,204 ZTF transients, 70% spectroscopically classified | `02_class_counts.png` |
| 2 | Cleaning found real problems: corrupted coordinates, duplicates, missing 2026 labels | *(table from notebook 1, §7–8)* |
| 3 | It's a magnitude-limited sample, so it favours bright objects | `02_cumulative_counts.png` |
| 4 | Galactic vs extragalactic: where things are on the sky | `02_sky_map.png` |
| 5 | Ia are standard candles | `02_peakabs_box.png`, `02_hubble_diagram.png` |
| 6 | Selection effects create fake trends | `02_malmquist_bias.png` |
| 7 | Ia have a timescale "sweet spot" | `03_ia_fraction_by_feature.png`, `02_rise_vs_fade.png` |
| 8 | The leakage trap: what you can know at detection time | `03_single_feature_auc.png` |
| 9 | Classifier results: purity vs completeness | `03_roc_pr_curves.png`, `03_threshold_tradeoff.png` |
| 10 | Honest validation: data drift since 2022 | `03_feature_drift.png` |
| 11 | The model doesn't matter much; the features set the ceiling. Leakage flatters every model | `04_model_comparison.png`, `04_leakage_by_model.png` |
| 12 | Neural networks: overfitting, seed luck, and a tie with gradient boosting | `04_nn_training_curves.png`, `04_nn_seed_spread.png` |
| 13 | Using the model under a capacity limit: safe daily volume depends on the class mix. Next steps | `05_safe_volume.png`, `05_monitoring.png` |

## Key findings

**Data quality**
* 33 rows had wrong galactic latitude `b` and extinction `A_V`; 23 of them share one identical value. Found by recomputing `b` from RA/Dec with astropy.
* 13 objects appear twice under different ZTF IDs (same IAU name and position).
* 2026 objects have almost no `type` values, even though ~2/3 already carry `SN2026…` names. The export lags behind the classifications.
* About 26% of rise times and 18% of fade times are lower limits (`>`). In 2022 the rate of rise-time limits jumped from ~5% to ~40%.

**Science**
* SN Ia are 68% of classified objects. They peak at M ≈ −19.2, with the narrowest spread of any class (middle 50% within 0.52 mag).
* Ia rise in ~10 days and fade by half in ~14 days. Fade times of 10–20 days are ~85–90% Ia.
* The catalogue cuts off at magnitude 19 and excludes |b| < 7°. This produces Malmquist bias: high-redshift Ia look artificially brighter.

**Machine learning**
* Baseline (always say "Ia"): 68% accuracy, AUC 0.50.
* Gradient boosting on photometric features: AUC 0.90, 86% purity at 94% completeness. Logistic regression reaches only 0.70, because the Ia pattern is a sweet spot rather than a straight line.
* Adding redshift-based features raises AUC to 0.98, but that's **leakage**: redshift comes from the classification spectrum itself.
* Predicting future years (train 2018–23, test 2024–25) gives AUC 0.85. Recent objects carry less information because many rise times are only limits.
* Across 14 models, every flexible one reaches CV AUC 0.87–0.89. Tuned, scikit-learn's gradient boosting, XGBoost and LightGBM are indistinguishable (CV 0.892–0.893, test 0.901–0.903); XGBoost's defaults overfit (0.870). Logistic regression with squared features jumps from 0.72 to 0.86: feature engineering beats model choice.
* Leakage adds +0.08 to +0.22 AUC to every model. About a third of logistic regression's gain comes only from *whether* redshift is missing (Galactic CVs and AGN have none).
* A tuned neural network (2 hidden layers, early stopping, 5-seed ensemble) reaches test AUC 0.895 vs 0.903 for tuned gradient boosting (a small but significant gap). On future years it's slightly ahead, 0.859 vs 0.854 (not significant). Changing the random seed alone moves its CV score by up to 0.011.
* Capacity scenario (333 candidates a day, 100 spectra, ≤ 10 false positives): taking the top 100 gives 96–97% precision and 6–7 false positives on the worst 5% of days at 68% Ia. On recent data, all 100 slots are safe only above about 65% Ia; at 50% Ia the safe volume is about 68 a day. The model estimates the full stream at about 67.5% Ia.

## Where to go next

1. **Full light curves** from ZTF alert brokers ([ALeRCE](https://alerce.science), [Fink](https://fink-broker.org), [Lasair](https://lasair-ztf.lsst.ac.uk)) or the ZTF forced-photometry service: colour, colour evolution and light-curve shape.
2. **Host-galaxy photometric redshifts** from galaxy catalogues: a leak-free replacement for redshift.
3. **Model tuning and probability calibration**, and **multi-class** classification (Ia / II / Ib/c / SLSN / …).
4. **Early classification** from the first few days of data, which is what spectroscopic follow-up decisions actually need.

## Data source

`bts.csv` is an export from the [ZTF Bright Transient Survey Sample Explorer](https://sites.astro.caltech.edu/ztf/bts/explorer.php). Background: Fremling et al. 2020 and Perley et al. 2020 (ApJ), which describe the BTS.
