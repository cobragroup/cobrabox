# Synthetic Data

This directory contains generated datasets for testing algorithms and running demonstrations. A table to navigate the available data can be found [here](https://docs.google.com/spreadsheets/d/1py2iG4EsWym2NCrDLYBB-f9zNlto4jUaQQdevKaV7wE/edit?gid=0#gid=0). 

Every `.csv` data file is accompanied by a metadata file named `info_[filename].json`.

## Dataset Structure

1. **Dummy Data** – Simplified datasets for basic feature and pipeline testing.
    * **Noise** – Various baseline noise models (Gaussian, 1/f, uncorrelated/correlated, synthetic seizures).  
      *Example contents:*
      * `dummy_noise_simulated_data_1.csv`: Independent, uncorrelated Gaussian noise.
      * `dummy_noise_simulated_data_3.csv`: Temporally independent Gaussian noise with block-structured spatial covariance.
      * `dummy_noise_simulated_data_4.csv`: Independent 1/f (pink) noise.
      * `dummy_noise_simulated_data_5.csv`: Independent Gaussian noise containing a synthetic seizure event. 
    * **Struct** – Vector Autoregressive (VAR) processes generated using specific connectivity matrices (chain, star, random) and autocorrelation settings. Includes both directed and symmetric topologies.

2. **Realistic Data** – Data simulated from VAR models fitted to actual patient iEEG data (Subject `ID1`, seizures `Sz13` and `Sz7` from the SWEC-ETHZ iEEG Database).  
    *Simulation Pipeline:*
    * The original data is divided into distinct clinical phases:    
        * *Interictal*: 3 to 1.5 minutes before the seizure mark.
        * *Preictal*: 1.5 to 0 minutes before the seizure mark.
        * *Onset*: The first 10 seconds of the seizure.
        * *Seizure*: The remainder of the seizure event.
    * For each phase, Principal Component Analysis (PCA) is applied for dimensionality reduction (retaining components that explain 95% of the data variance).
    * A VAR model is estimated and simulated within this latent PCA space.
    * An inverse PCA transformation is applied to project the simulated signals back into the original iEEG sensor space.

3. **Biorealistic Data** – This dataset was generated using the *Wendling neural mass model*, which simulates the transition from interictal to ictal states in human temporal lobe epilepsy. For full details on the underlying computational framework, please refer to the original study by [Wendling et al. (2005)](https://doi.org/10.1097/01.wnp.0000184051.37267.f0).