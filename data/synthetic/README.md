# Synthetic Data

This directory contains generated datasets for testing algorithms and running demonstrations. A table to navigate the available data can be found [here](https://docs.google.com/spreadsheets/d/1py2iG4EsWym2NCrDLYBB-f9zNlto4jUaQQdevKaV7wE/edit?gid=0#gid=0).

Every `.csv` data file is accompanied by a metadata file named `info_[filename].json`.

## Dataset Structure

The dummy and realistic datasets that used to live here were removed: cobrabox
no longer ships data with the package. The demo dataset is now `swiss_demo`,
loaded with `cb.load_dataset("swiss_demo")` and downloaded on first use.

1. **Biorealistic Data** – This dataset was generated using the *Wendling neural mass model*, which simulates the transition from interictal to ictal states in human temporal lobe epilepsy. For full details on the underlying computational framework, please refer to the original study by [Wendling et al. (2005)](https://doi.org/10.1097/01.wnp.0000184051.37267.f0).
