# Windowing & Aggregation

*How do I analyze temporal dynamics?*

Features in the `cobrabox.windowing` domain. Access them as `cb.windowing.<Feature>` or `cb.feature.<Feature>`.

### ConcatAggregate
Aggregate a stream of per-window Data by stacking along a new 'window' dimension.

**Tags:** [`aggregation`](../tags/aggregation.md), [`io:preserves-time`](../tags/io-preserves-time.md)

### MeanAggregate
Aggregate a stream of per-window Data by averaging across windows.

**Tags:** [`aggregation`](../tags/aggregation.md), [`io:scalar-per-channel`](../tags/io-scalar-per-channel.md)

### SlidingWindow
Yield one Data per sliding window over the time dimension.

**Tags:** [`temporal-dynamics`](../tags/temporal-dynamics.md), [`segmentation`](../tags/segmentation.md), [`nonstationarity`](../tags/nonstationarity.md), [`io:iterator`](../tags/io-iterator.md)

### SlidingWindowReduce
Sliding window with automatic per-window reduction.

**Tags:** [`temporal-dynamics`](../tags/temporal-dynamics.md), [`segmentation`](../tags/segmentation.md), [`reduction`](../tags/reduction.md), [`nonstationarity`](../tags/nonstationarity.md), [`io:preserves-time`](../tags/io-preserves-time.md)
