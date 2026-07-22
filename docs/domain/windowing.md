# Windowing & Aggregation

*How do I analyze temporal dynamics?*

Features in the `cobrabox.windowing` domain. Access them as `cb.<Feature>` (canonical), or as `cb.windowing.<Feature>` / `cb.feature.<Feature>`.

### ConcatAggregate
Aggregate a stream of per-window Data by stacking along a new 'window' dimension.

**Tags:** [`aggregation`](../tags.md#tag-aggregation), [`io:preserves-time`](../tags.md#tag-io-preserves-time)

### MeanAggregate
Aggregate a stream of per-window Data by averaging across windows.

**Tags:** [`aggregation`](../tags.md#tag-aggregation), [`io:scalar-per-channel`](../tags.md#tag-io-scalar-per-channel)

### SlidingWindow
Yield one Data per sliding window over the time dimension.

**Tags:** [`temporal-dynamics`](../tags.md#tag-temporal-dynamics), [`segmentation`](../tags.md#tag-segmentation), [`nonstationarity`](../tags.md#tag-nonstationarity), [`io:iterator`](../tags.md#tag-io-iterator)

### SlidingWindowReduce
Sliding window with automatic per-window reduction.

**Tags:** [`temporal-dynamics`](../tags.md#tag-temporal-dynamics), [`segmentation`](../tags.md#tag-segmentation), [`reduction`](../tags.md#tag-reduction), [`nonstationarity`](../tags.md#tag-nonstationarity), [`io:preserves-time`](../tags.md#tag-io-preserves-time)
