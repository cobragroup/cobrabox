# Synthetic Data

Generated datasets for testing and demos. Magic table to orient in the data is [here](https://docs.google.com/spreadsheets/d/1py2iG4EsWym2NCrDLYBB-f9zNlto4jUaQQdevKaV7wE/edit?gid=0#gid=0)

Simulated data structure:
1. **Dummy data** - easy data for basic feature testing
    - *Noise* - different types of noise (Gaussian, 1/f, uncorrelated/correlated, synthetic seizures)
    - *Struct* - vectore autoregressive process (VAR) with different connectivity matrices and autocorrelation. Chain, star, random connectivity; directed or symmetric.
2. **Realistic data** - VAR process fit to iEEG data (subject ID1, seizures Sz13 and Sz7 from The SWEC-ETHZ iEEG Database)
3. **Biorealistic data** - multi-nodes version of Wendling model (Wendling, Fabrice*; Hernandez, Alfredo*; Bellanger, Jean-Jacques*; Chauvel, Patrick†; Bartolomei, Fabrice†. Interictal to Ictal Transition in Human Temporal Lobe Epilepsy: Insights From a Computational Model of Intracerebral EEG. Journal of Clinical Neurophysiology 22(5):p 343-356, October 2005. | DOI: 110.1097/01.wnp.0000184051.37267.f0 )

Example contents:

- `toy_eeg.npy`
- `toy_fmri.npy`
