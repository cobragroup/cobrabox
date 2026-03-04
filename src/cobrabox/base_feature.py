from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

import xarray as xr

from .data import Data


@dataclass
class BaseFeature(ABC):
    """Base class for all cobrabox features (Data → Data).

    Features are dataclasses that store configuration at initialization
    and implement transform logic in __call__.
    """

    _is_cobrabox_feature: ClassVar[bool] = True

    def __or__(self, other: BaseFeature) -> Pipeline:
        """Enable pipe syntax: Feature1() | Feature2()"""
        return Pipeline(self, other)

    @abstractmethod
    def __call__(self, data: Data) -> xr.DataArray | Data:
        """Apply the feature transformation. Subclasses implement this."""

    def apply(self, data: Data) -> Data:
        """Apply feature and wrap result in Data with history tracking."""
        result = self(data)
        if not isinstance(result, (xr.DataArray, Data)):
            raise TypeError(
                f"Feature '{self.__class__.__name__}' must return "
                f"xarray.DataArray or Data, got {type(result)}"
            )
        return data._copy_with_new_data(new_data=result, operation_name=self.__class__.__name__)


@dataclass
class SplitterFeature(ABC):
    """Base class for features that split one Data into a stream of Data (Data → Iterator[Data]).

    Used for windowing operations. Subclasses implement __call__ as a generator.
    Supports | to begin building a Chord:

        SlidingWindow(10, 5) | LineLength() | MeanAggregate()  # → Chord
    """

    _is_cobrabox_feature: ClassVar[bool] = True

    @abstractmethod
    def __call__(self, data: Data) -> Iterator[Data]:
        """Yield one Data object per split (e.g. per window)."""

    def __or__(self, other: BaseFeature | AggregatorFeature) -> _ChordBuilder | Chord:
        """Start building a Chord: SplitterFeature | pipeline_step | AggregatorFeature."""
        if isinstance(other, AggregatorFeature):
            raise TypeError(
                "Cannot pipe a SplitterFeature directly into an AggregatorFeature — "
                "add at least one pipeline step in between."
            )
        return _ChordBuilder(split=self, pipeline=other)


@dataclass
class AggregatorFeature(ABC):
    """Base class for features that fold a stream of Data into one (Iterable[Data] → Data).

    Used after windowed processing to combine per-window results.
    """

    _is_cobrabox_feature: ClassVar[bool] = True

    @abstractmethod
    def __call__(self, data: Data, stream: Iterator[Data]) -> Data:
        """Aggregate a stream of Data into a single Data.

        Args:
            data: Original Data before splitting (for metadata/history).
            stream: Iterator of per-window processed Data objects.
        """


class _ChordBuilder:
    """Intermediate object produced by SplitterFeature | pipeline_step.

    Continue piping to extend the inner pipeline, or pipe into an
    AggregatorFeature to finalise the Chord:

        SlidingWindow(10, 5) | LineLength() | BandpassFilter() | MeanAggregate()
        #                                                        ↑ finalises here
    """

    def __init__(self, split: SplitterFeature, pipeline: BaseFeature | Pipeline) -> None:
        self.split = split
        self.pipeline = pipeline

    def __or__(self, other: BaseFeature | AggregatorFeature) -> _ChordBuilder | Chord:
        if isinstance(other, AggregatorFeature):
            return Chord(split=self.split, pipeline=self.pipeline, aggregate=other)
        # Extend the inner pipeline
        if isinstance(self.pipeline, Pipeline):
            new_pipeline = self.pipeline | other
        else:
            new_pipeline = Pipeline(self.pipeline, other)
        return _ChordBuilder(split=self.split, pipeline=new_pipeline)

    def apply(self, data: Data) -> Data:
        raise TypeError(
            "_ChordBuilder is incomplete — pipe into an AggregatorFeature to finalise the Chord.\n"
            "Example: SlidingWindow() | LineLength() | MeanAggregate()"
        )


@dataclass
class Chord(BaseFeature):
    """Fan-out → map → fan-in composition.

    Splits input data with a SplitterFeature, applies a pipeline to each
    split lazily, then aggregates all results with an AggregatorFeature.

    Can be constructed explicitly or via pipe syntax:

        SlidingWindow(10, 5) | LineLength() | MeanAggregate()

    Example:
        >>> chord = SlidingWindow(100, 50) | BandpassFilter() | MeanAggregate()
        >>> result = chord.apply(data)
    """

    split: SplitterFeature
    pipeline: BaseFeature | Pipeline
    aggregate: AggregatorFeature

    def __call__(self, data: Data) -> Data:
        stream = self.split(data)
        processed = (self.pipeline.apply(w) for w in stream)
        return self.aggregate(data, processed)


class Pipeline(list[BaseFeature]):
    """Sequential pipeline of features using pipe syntax."""

    def __init__(self, *features: BaseFeature) -> None:
        super().__init__(features)
        self.features = features

    def __or__(self, other: BaseFeature) -> Pipeline:
        """Enable chaining: pipeline | AnotherFeature()"""
        return Pipeline(*self.features, other)

    def apply(self, data: Data) -> Data:
        """Apply all features in sequence."""
        result = data
        for feature in self.features:
            result = feature.apply(result)
        return result

    def __call__(self, data: Data) -> Data:
        """Allow pipeline(data) syntax."""
        return self.apply(data)
