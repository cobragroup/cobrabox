# Examples

Minimal runnable scripts. Run from the project root with `uv run python examples/<script>.py`.

| Script | What it shows |
| ------ | ------------- |
| `data_basics.py` | Loading datasets, accessing dimensions and coordinates, creating Data from numpy, filtering and grouping |
| `creating_dummy_data.py` | Creating EEG data from numpy and running a feature pipeline |
| `dataset_loader_demo.py` | Loading a dataset and inspecting shapes and metadata |
| `realistic_dataset_loader_demo.py` | Loading a realistic dataset with subject metadata |
| `feature_pipeline_demo.py` | Building and running feature pipelines |
| `feature_phase_locking_value_demo.py` | Phase locking value connectivity feature |
| `partial_correlation_demo.py` | Partial correlation connectivity feature |
| `sliding_window_aggregators_demo.py` | SlidingWindow with MeanAggregate (reduces windows) vs ConcatAggregate (preserves windows) |
