# API Reference

Auto-generated API documentation for all CobraBox modules.

## Core Modules

- [Data](data.md) - Data containers (`Data`, `EEG`, `FMRI`)
- [Features](features.md) - Feature functions
- [Datasets](datasets.md) - Dataset collection (`Dataset[T]`) and loading
- Serialization - `cb.save`, `cb.load`, `cb.serialize`, `cb.deserialize` (see [Pipelines guide](../guide/pipelines.md#serialization))

## Quick Links

- [`Data`](data.md#cobrabox.data.Data) - Main data container
- [`EEG`](data.md#cobrabox.data.EEG) - EEG data subclass
- [`FMRI`](data.md#cobrabox.data.FMRI) - fMRI data subclass
- [`from_numpy`](data.md#cobrabox.data.Data.from_numpy) - Create from numpy array
- [`from_xarray`](data.md#cobrabox.data.Data.from_xarray) - Create from xarray
- [`feature.*`](features.md) - All feature functions
- [`Dataset`](datasets.md#cobraboxdatasetDataset) - Typed collection of Data objects
- [`load_dataset()`](datasets.md#cobrabox.datasets.load_dataset) - Load datasets (local and remote)
- `cb.save(obj, path)` / `cb.load(path)` - Pipeline file I/O
- `cb.serialize(obj)` / `cb.deserialize(content)` - Pipeline string I/O
