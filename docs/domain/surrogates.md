# Surrogates

*How do I test statistical significance?*

Features in the `cobrabox.surrogates` domain. Access them as `cb.<Feature>` (canonical), or as `cb.surrogates.<Feature>` / `cb.feature.<Feature>`.

Each feature has two forms: a **class** for building pipelines, and a one-shot **function** for a single call.

### FourierTransformSurrogates
Generate Fourier transform surrogates (preserving autocorrelation) of SignalData.

```python
cb.fourier_transform_surrogates(data, ...)  # one-shot
cb.FourierTransformSurrogates(...).apply(data)  # composable, for pipelines
```

**Tags:** [`null-hypothesis`](../tags.md#tag-null-hypothesis), [`nonlinearity-test`](../tags.md#tag-nonlinearity-test), [`phase-randomization`](../tags.md#tag-phase-randomization), [`autocorrelation-preserving`](../tags.md#tag-autocorrelation-preserving), [`surrogate-significance`](../tags.md#tag-surrogate-significance), [`epilepsy`](../tags.md#tag-epilepsy), [`io:iterator`](../tags.md#tag-io-iterator)
