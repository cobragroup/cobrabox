# CobraBox – Tutorials & Example Pipelines

*[COBRA group](https://cobra.cs.cas.cz), Institute of Computer Science, The Czech Academy of Sciences*

[CobraBox](https://github.com/cobragroup/cobrabox) is a Python toolbox for the analysis of multivariate brain time-series, such as EEG, intracranial EEG (iEEG) and fMRI. It stores signals in labelled data objects built on [xarray](https://docs.xarray.dev/en/stable/) (e.g. `SignalData` and `Dataset`), so every array carries named dimensions such as `time` and `space` together with metadata like the sampling rate and subject ID. On top of these objects, CobraBox provides signal processing (filtering, power spectra, band power), connectivity and network measures (e.g. Partial Directed Coherence, Directed Transfer Function, Reciprocal Connectivity), and loaders that fetch curated public datasets directly into the CobraBox format.

This folder is the entry point for getting started with CobraBox. It contains a tutorial and worked examples in the form of Jupyter notebooks.

&nbsp;
## What you will find here

```
.
├── README.md
├── tutorial_data_exploration.ipynb          # Tutorial: exploring and fetching remote datasets
├── seizure_detection_minimal.ipynb          # Use-case: seizure detection
├── 01_preprocessing_and_computations.ipynb  # Use-case: directed connectivity, part 1
├── 02_analysis_stats_and_ML.ipynb           # Use-case: directed connectivity, part 2
├── local_utils.py                           # Helper module for the two directed-connectivity notebooks
```

### Tutorial notebook

**[Exploring and Fetching Remote Datasets](tutorial_data_exploration.ipynb)** is the recommended starting point. It introduces the CobraBox data objects (`Dataset` and `SignalData`), their xarray backbone, their dimensions and attributes, and how to access the numeric data. It shows how to list the curated datasets available in CobraBox (`cb.list_datasets()`, `cb.show_datasets()`), inspect a dataset before downloading it (`cb.dataset_info()`), and fetch it from an open repository (`cb.load_dataset()`). Using one subject of the Zurich interictal iEEG dataset from OpenNeuro, it then walks through first preprocessing steps: plotting raw signals, 50 Hz notch filtering, low-pass filtering, power spectra, band power and the bipolar montage.

### Use-cases and worked examples

Use-cases show how CobraBox is combined into a complete analysis on real data.

- **[Seizure detection](seizure_detection_minimal.ipynb)**: A minimal example pipeline for seizure detection. *(Short description to be added.)*

- **Directed connectivity for localizing epileptogenicity**: A two-part, reproducible pipeline on the 20 patients of the [Zurich interictal iEEG dataset](https://openneuro.org/datasets/ds003498/versions/1.1.1), following Stergiadis et al. (2026). The two notebooks must be run in order.
  1. **[Part 1: Data preparation and connectivity estimation](01_preprocessing_and_computations.ipynb)**: Loads the clean version of the dataset, applies a 50 Hz notch filter and a single-spacing bipolar montage, and extracts 20 random 3-second segments per subject. It then computes directed connectivity with the Directed Transfer Function (`cb.feature.DirectedTransferFunction`), averages it within eight frequency bands (delta to fast ripples), and derives the inward and outward strength of each electrode contact. All results are saved to `data/`.
  2. **[Part 2: Analysis, statistics and machine learning](02_analysis_stats_and_ML.ipynb)**: Uses the results saved by Part 1 to compare inward and outward strength inside vs. outside the surgical resection in good- and poor-outcome patients (paired Wilcoxon tests with FDR correction). It then trains a logistic-regression classifier to identify resected contacts, and predicts the surgical outcome at patient level.

  Both notebooks import `local_utils.py`, which holds routine steps that are not part of CobraBox itself: reading the patient metadata, notch filtering, building the bipolar montage, selecting segments, and saving/loading intermediate files. The helpers are always called explicitly as `local_utils.<function>()`, so you can open the module to see exactly what each step does.

&nbsp;
## Requirements

To run the notebooks you need:

- **Python** ≥ 3.11
- **CobraBox** ≥ 1.0, which installs its core dependencies (**NumPy** ≥ 2.4.2, **xarray** > 2026.2.0):
```
  pip install cobrabox
```
- **Matplotlib** for the figures (all notebooks).
- For the directed-connectivity use-case, additionally **pandas**, **SciPy**, **scikit-learn** and **tqdm**:
```
  pip install matplotlib pandas scipy scikit-learn tqdm
```

**Data and disk space.** The notebooks download data automatically from OpenNeuro into a local `data/` folder (set with `cb.set_dataset_dir()`). The full Zurich iEEG dataset is about 60 GB (roughly 3 GB per subject). The tutorial downloads a single subject; Part 1 of the directed-connectivity use-case downloads all 20 subjects, so make sure you have enough free space. When a download starts, CobraBox shows its size and asks for confirmation (type `y`).

**Patient metadata.** The directed-connectivity notebooks read the surgical outcome and resected channels of each patient from a CSV file in `data/` (see `local_utils.load_patient_info()`).

&nbsp;
## Further documentation

- CobraBox source code and issue tracker: <https://github.com/cobragroup/cobrabox>
- CobraBox documentation and installation instructions: *(link to be added)*

&nbsp;
## References and citation

The directed-connectivity use-case follows:

> Stergiadis C, Halliday DM, Kazis D, Klados MA. High-frequency directed networks can identify epileptogenic tissue and predict surgical outcome in drug-resistant epilepsy. *Epilepsy Research*. 2026;226:107838. doi: [10.1016/j.eplepsyres.2026.107838](https://doi.org/10.1016/j.eplepsyres.2026.107838)

The Zurich interictal iEEG dataset is available on OpenNeuro: [ds003498](https://openneuro.org/datasets/ds003498/versions/1.1.1).
