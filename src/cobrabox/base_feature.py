from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

import xarray as xr

from .data import Data

# Type variable for generic features - allows subclasses to specify Data or SignalData
DataT = TypeVar("DataT", bound=Data)


@dataclass
class BaseFeature(ABC, Generic[DataT]):
    """Base class for all cobrabox features (Data → Data).

    Features are dataclasses that store configuration at initialization
    and implement transform logic in __call__.

    Type Parameters:
        DataT: The type of data this feature accepts. Defaults to Data (any data),
            but subclasses can narrow to SignalData for time-series features.
    """

    _is_cobrabox_feature: ClassVar[bool] = True
    output_type: ClassVar[type[Data] | None] = None
    """Output container type. None means same as input, Data means plain Data,
    SignalData means SignalData with time dimension."""

    def __or__(self, other: BaseFeature[DataT]) -> Pipeline[DataT]:
        """Enable pipe syntax: Feature1() | Feature2()"""
        return Pipeline(self, other)

    @abstractmethod
    def __call__(self, data: DataT) -> xr.DataArray | Data:
        """Apply the feature transformation. Subclasses implement this."""

    def apply(self, data: DataT) -> Data:
        """Apply feature and wrap result in Data with history tracking.

        Returns:
            Data: The output container. If output_type is set, returns that type.
                If output_type is None, returns the same type as input.
        """
        result = self(data)
        if not isinstance(result, (xr.DataArray, Data)):
            raise TypeError(
                f"Feature '{self.__class__.__name__}' must return "
                f"xarray.DataArray or Data, got {type(result)}"
            )
        # Determine output container class (None means same as input)
        output_cls = self.output_type if self.output_type is not None else type(data)
        return output_cls._copy_with_new_data(
            data, new_data=result, operation_name=self.__class__.__name__
        )


@dataclass
class SplitterFeature(ABC, Generic[DataT]):
    """Base class for features that split one Data into a stream of Data (Data → Iterator[Data]).

    Used for windowing operations. Subclasses implement __call__ as a generator.
    Supports | to begin building a Chord:

        SlidingWindow(10, 5) | LineLength() | MeanAggregate()  # → Chord

    Type Parameters:
        DataT: The type of data this feature accepts. Defaults to Data (any data),
            but subclasses can narrow to SignalData for time-series features.
    """

    _is_cobrabox_feature: ClassVar[bool] = True

    @abstractmethod
    def __call__(self, data: DataT) -> Iterator[Data]:
        """Yield one Data object per split (e.g. per window)."""

    def __or__(self, other: BaseFeature[Any]) -> _ChordBuilder[DataT]:
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


class _ChordBuilder(Generic[DataT]):
    """Intermediate object produced by SplitterFeature | pipeline_step.

    Continue piping to extend the inner pipeline, or pipe into an
    AggregatorFeature to finalise the Chord:

        SlidingWindow(10, 5) | LineLength() | BandpassFilter() | MeanAggregate()
        #                                                        ↑ finalises here
    """

    def __init__(
        self, split: SplitterFeature[DataT], pipeline: BaseFeature[Any] | Pipeline[Any]
    ) -> None:
        self.split = split
        self.pipeline = pipeline

    def __or__(
        self, other: BaseFeature[Any] | AggregatorFeature
    ) -> _ChordBuilder[DataT] | Chord[DataT]:
        if isinstance(other, AggregatorFeature):
            return Chord(split=self.split, pipeline=self.pipeline, aggregate=other)
        # Extend the inner pipeline
        if isinstance(self.pipeline, Pipeline):
            new_pipeline = self.pipeline | other
        else:
            new_pipeline = Pipeline(self.pipeline, other)
        return _ChordBuilder(split=self.split, pipeline=new_pipeline)

    def apply(self, data: DataT) -> Data:
        # TODO: rewrite to more helpful and use-friendly errors, users might be stupid
        raise TypeError(
            "_ChordBuilder is incomplete — pipe into an AggregatorFeature to finalise the Chord.\n"
            "Example: SlidingWindow() | LineLength() | MeanAggregate()"
        )


@dataclass
class Chord(BaseFeature[DataT]):
    """Fan-out → map → fan-in composition.

    Splits input data with a SplitterFeature, applies a pipeline to each
    split lazily, then aggregates all results with an AggregatorFeature.

    Can be constructed explicitly or via pipe syntax:

        SlidingWindow(10, 5) | LineLength() | MeanAggregate()

    Example:
        >>> chord = SlidingWindow(100, 50) | BandpassFilter() | MeanAggregate()
        >>> result = chord.apply(data)

    Type Parameters:
        DataT: The type of data this chord accepts. Inherited from the splitter.
    """

    split: SplitterFeature[DataT]
    pipeline: BaseFeature[Any] | Pipeline[Any]
    aggregate: AggregatorFeature

    def __call__(self, data: DataT) -> Data:
        stream = self.split(data)
        processed = (self.pipeline.apply(w) for w in stream)
        return self.aggregate(data, processed)


class Pipeline(list[BaseFeature[DataT]], Generic[DataT]):
    """Sequential pipeline of features using pipe syntax.

    Type Parameters:
        DataT: The type of data this pipeline accepts.
    """

    def __init__(self, *features: BaseFeature[DataT]) -> None:
        super().__init__(features)
        self.features = features

    def __or__(self, other: BaseFeature[DataT]) -> Pipeline[DataT]:
        """Enable chaining: pipeline | AnotherFeature()"""
        return Pipeline(*self.features, other)

    def apply(self, data: DataT) -> Data:
        """Apply all features in sequence."""
        result: Data = data
        for feature in self.features:
            result = feature.apply(result)  # type: ignore[arg-type]
        return result

    def __call__(self, data: DataT) -> Data:
        """Allow pipeline(data) syntax."""
        return self.apply(data)
