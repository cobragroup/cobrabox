# Tags

Cross-cutting discovery across domain boundaries. Every feature is tagged by method, modality, application, IO shape and requirements. Each tag below lists the features that carry it.

## Modality

<a id="tag-ecg"></a>
`ecg` (1) — [SampleEntropy](domain/infometrics.md)

<a id="tag-eeg"></a>
`eeg` (32) — [AmplitudeEntropy](domain/infometrics.md), [AmplitudeVariation](domain/signalstats.md), [AnalyticSignal](domain/transforms.md), [Autocorrelation](domain/signalstats.md), [BandPower](domain/spectral.md), [BandpassFilter](domain/transforms.md), [Coherence](domain/connectivity.md), [ContinuousWaveletTransform](domain/spectral.md), [Cordance](domain/spectral.md), [Correlation](domain/connectivity.md), [Covariance](domain/connectivity.md), [DirectedTransferFunction](domain/connectivity.md), [DiscreteWaveletTransform](domain/spectral.md), [EMD](domain/decompositions.md), [EnvelopeCorrelation](domain/connectivity.md), [EpileptogenicityIndex](domain/signalstats.md), [FourierTransform](domain/transforms.md), [FractalDimension](domain/infometrics.md), [GrangerCausality](domain/connectivity.md), [LempelZiv](domain/infometrics.md), [LineLength](domain/signalstats.md), [MutualInformation](domain/connectivity.md), [Nonreversibility](domain/infometrics.md), [PartialCorrelation](domain/connectivity.md), [PartialDirectedCoherence](domain/connectivity.md), [PhaseLockingValue](domain/connectivity.md), [ReciprocalConnectivity](domain/connectivity.md), [RecurrenceMatrix](domain/infometrics.md), [SVD](domain/decompositions.md), [SampleEntropy](domain/infometrics.md), [Spectrogram](domain/spectral.md), [SpikeCount](domain/signalstats.md)

<a id="tag-fmri"></a>
`fmri` (8) — [AnalyticSignal](domain/transforms.md), [Autocorrelation](domain/signalstats.md), [BandPower](domain/spectral.md), [Correlation](domain/connectivity.md), [Covariance](domain/connectivity.md), [FourierTransform](domain/transforms.md), [PartialCorrelation](domain/connectivity.md), [Spectrogram](domain/spectral.md)

<a id="tag-intracranial-eeg"></a>
`intracranial-eeg` (1) — [EpileptogenicityIndex](domain/signalstats.md)

<a id="tag-meg"></a>
`meg` (1) — [EnvelopeCorrelation](domain/connectivity.md)

## Application domain

<a id="tag-aging"></a>
`aging` (1) — [FractalDimension](domain/infometrics.md)

<a id="tag-anesthesia"></a>
`anesthesia` (6) — [AmplitudeVariation](domain/signalstats.md), [BandPower](domain/spectral.md), [FractalDimension](domain/infometrics.md), [LempelZiv](domain/infometrics.md), [PhaseLockingValue](domain/connectivity.md), [SampleEntropy](domain/infometrics.md)

<a id="tag-artifact"></a>
`artifact` (1) — [SpikeCount](domain/signalstats.md)

<a id="tag-consciousness"></a>
`consciousness` (1) — [LempelZiv](domain/infometrics.md)

<a id="tag-dementia"></a>
`dementia` (1) — [FractalDimension](domain/infometrics.md)

<a id="tag-depression"></a>
`depression` (2) — [BandPower](domain/spectral.md), [Cordance](domain/spectral.md)

<a id="tag-epilepsy"></a>
`epilepsy` (15) — [BandPower](domain/spectral.md), [Coherence](domain/connectivity.md), [DirectedTransferFunction](domain/connectivity.md), [EpileptogenicityIndex](domain/signalstats.md), [FourierTransformSurrogates](domain/surrogates.md), [GrangerCausality](domain/connectivity.md), [LempelZiv](domain/infometrics.md), [LineLength](domain/signalstats.md), [Nonreversibility](domain/infometrics.md), [PartialDirectedCoherence](domain/connectivity.md), [PhaseLockingValue](domain/connectivity.md), [ReciprocalConnectivity](domain/connectivity.md), [RecurrenceMatrix](domain/infometrics.md), [SampleEntropy](domain/infometrics.md), [SpikeCount](domain/signalstats.md)

<a id="tag-heart-rate-variability"></a>
`heart-rate-variability` (1) — [SampleEntropy](domain/infometrics.md)

<a id="tag-qeeg"></a>
`qeeg` (1) — [Cordance](domain/spectral.md)

<a id="tag-seizure-detection"></a>
`seizure-detection` (1) — [LineLength](domain/signalstats.md)

<a id="tag-seizure-onset"></a>
`seizure-onset` (2) — [ContinuousWaveletTransform](domain/spectral.md), [Spectrogram](domain/spectral.md)

<a id="tag-seizure-onset-zone"></a>
`seizure-onset-zone` (2) — [EpileptogenicityIndex](domain/signalstats.md), [ReciprocalConnectivity](domain/connectivity.md)

<a id="tag-seizure-propagation"></a>
`seizure-propagation` (3) — [DirectedTransferFunction](domain/connectivity.md), [GrangerCausality](domain/connectivity.md), [PartialDirectedCoherence](domain/connectivity.md)

<a id="tag-sleep"></a>
`sleep` (1) — [BandPower](domain/spectral.md)

<a id="tag-spike-detection"></a>
`spike-detection` (1) — [SpikeCount](domain/signalstats.md)

<a id="tag-treatment-response"></a>
`treatment-response` (1) — [Cordance](domain/spectral.md)

## Research paradigm

<a id="tag-default-mode"></a>
`default-mode` (1) — [Correlation](domain/connectivity.md)

<a id="tag-erps"></a>
`erps` (1) — [ContinuousWaveletTransform](domain/spectral.md)

<a id="tag-event-related"></a>
`event-related` (1) — [Spectrogram](domain/spectral.md)

<a id="tag-nonstationarity"></a>
`nonstationarity` (2) — [SlidingWindow](domain/windowing.md), [SlidingWindowReduce](domain/windowing.md)

<a id="tag-resting-state"></a>
`resting-state` (8) — [Coherence](domain/connectivity.md), [Correlation](domain/connectivity.md), [Covariance](domain/connectivity.md), [EnvelopeCorrelation](domain/connectivity.md), [GrangerCausality](domain/connectivity.md), [MutualInformation](domain/connectivity.md), [PartialCorrelation](domain/connectivity.md), [PhaseLockingValue](domain/connectivity.md)

## Connectivity

<a id="tag-aec"></a>
`aec` (1) — [EnvelopeCorrelation](domain/connectivity.md)

<a id="tag-causality"></a>
`causality` (1) — [Nonreversibility](domain/infometrics.md)

<a id="tag-conditional-independence"></a>
`conditional-independence` (1) — [PartialCorrelation](domain/connectivity.md)

<a id="tag-cross-spectral"></a>
`cross-spectral` (1) — [Coherence](domain/connectivity.md)

<a id="tag-direct-causality"></a>
`direct-causality` (1) — [PartialDirectedCoherence](domain/connectivity.md)

<a id="tag-directed"></a>
`directed` (4) — [DirectedTransferFunction](domain/connectivity.md), [GrangerCausality](domain/connectivity.md), [PartialDirectedCoherence](domain/connectivity.md), [ReciprocalConnectivity](domain/connectivity.md)

<a id="tag-functional-connectivity"></a>
`functional-connectivity` (1) — [RecurrenceMatrix](domain/infometrics.md)

<a id="tag-phase-synchrony"></a>
`phase-synchrony` (1) — [PhaseLockingValue](domain/connectivity.md)

<a id="tag-sink-source"></a>
`sink-source` (1) — [ReciprocalConnectivity](domain/connectivity.md)

<a id="tag-total-causality"></a>
`total-causality` (1) — [DirectedTransferFunction](domain/connectivity.md)

<a id="tag-undirected"></a>
`undirected` (7) — [Coherence](domain/connectivity.md), [Correlation](domain/connectivity.md), [Covariance](domain/connectivity.md), [EnvelopeCorrelation](domain/connectivity.md), [MutualInformation](domain/connectivity.md), [PartialCorrelation](domain/connectivity.md), [PhaseLockingValue](domain/connectivity.md)

<a id="tag-volume-conduction"></a>
`volume-conduction` (1) — [EnvelopeCorrelation](domain/connectivity.md)

## Method

<a id="tag-autocorrelation-preserving"></a>
`autocorrelation-preserving` (1) — [FourierTransformSurrogates](domain/surrogates.md)

<a id="tag-box-counting"></a>
`box-counting` (1) — [FractalDimension](domain/infometrics.md)

<a id="tag-butterworth"></a>
`butterworth` (1) — [BandpassFilter](domain/transforms.md)

<a id="tag-compression"></a>
`compression` (2) — [DiscreteWaveletTransform](domain/spectral.md), [LempelZiv](domain/infometrics.md)

<a id="tag-energy-ratio"></a>
`energy-ratio` (1) — [EpileptogenicityIndex](domain/signalstats.md)

<a id="tag-entropy"></a>
`entropy` (2) — [MutualInformation](domain/connectivity.md), [SampleEntropy](domain/infometrics.md)

<a id="tag-entropy-production"></a>
`entropy-production` (1) — [Nonreversibility](domain/infometrics.md)

<a id="tag-fft"></a>
`fft` (1) — [FourierTransform](domain/transforms.md)

<a id="tag-fractal"></a>
`fractal` (1) — [FractalDimension](domain/infometrics.md)

<a id="tag-higuchi"></a>
`higuchi` (1) — [FractalDimension](domain/infometrics.md)

<a id="tag-hilbert"></a>
`hilbert` (1) — [AnalyticSignal](domain/transforms.md)

<a id="tag-histogram"></a>
`histogram` (1) — [AmplitudeEntropy](domain/infometrics.md)

<a id="tag-ifft"></a>
`ifft` (1) — [InverseFourierTransform](domain/transforms.md)

<a id="tag-katz"></a>
`katz` (1) — [FractalDimension](domain/infometrics.md)

<a id="tag-kl-divergence"></a>
`kl-divergence` (1) — [Nonreversibility](domain/infometrics.md)

<a id="tag-mvar"></a>
`mvar` (4) — [DirectedTransferFunction](domain/connectivity.md), [GrangerCausality](domain/connectivity.md), [Nonreversibility](domain/infometrics.md), [PartialDirectedCoherence](domain/connectivity.md)

<a id="tag-nonlinearity-test"></a>
`nonlinearity-test` (1) — [FourierTransformSurrogates](domain/surrogates.md)

<a id="tag-orthogonalization"></a>
`orthogonalization` (1) — [EnvelopeCorrelation](domain/connectivity.md)

<a id="tag-page-hinkley"></a>
`page-hinkley` (1) — [EpileptogenicityIndex](domain/signalstats.md)

<a id="tag-pearson"></a>
`pearson` (1) — [Correlation](domain/connectivity.md)

<a id="tag-phase-randomization"></a>
`phase-randomization` (1) — [FourierTransformSurrogates](domain/surrogates.md)

<a id="tag-precision-matrix"></a>
`precision-matrix` (1) — [PartialCorrelation](domain/connectivity.md)

<a id="tag-prediction-error"></a>
`prediction-error` (1) — [GrangerCausality](domain/connectivity.md)

<a id="tag-shannon-entropy"></a>
`shannon-entropy` (1) — [AmplitudeEntropy](domain/infometrics.md)

<a id="tag-spearman"></a>
`spearman` (1) — [Correlation](domain/connectivity.md)

<a id="tag-stft"></a>
`stft` (1) — [Spectrogram](domain/spectral.md)

<a id="tag-surrogate-significance"></a>
`surrogate-significance` (1) — [FourierTransformSurrogates](domain/surrogates.md)

<a id="tag-var"></a>
`var` (3) — [DirectedTransferFunction](domain/connectivity.md), [GrangerCausality](domain/connectivity.md), [PartialDirectedCoherence](domain/connectivity.md)

<a id="tag-wavelet"></a>
`wavelet` (2) — [ContinuousWaveletTransform](domain/spectral.md), [DiscreteWaveletTransform](domain/spectral.md)

<a id="tag-welch"></a>
`welch` (2) — [BandPower](domain/spectral.md), [Coherence](domain/connectivity.md)

## Signal representation

<a id="tag-absolute-power"></a>
`absolute-power` (1) — [Cordance](domain/spectral.md)

<a id="tag-alpha"></a>
`alpha` (1) — [BandPower](domain/spectral.md)

<a id="tag-beta"></a>
`beta` (1) — [BandPower](domain/spectral.md)

<a id="tag-beta-gamma"></a>
`beta-gamma` (1) — [EpileptogenicityIndex](domain/signalstats.md)

<a id="tag-delta"></a>
`delta` (1) — [BandPower](domain/spectral.md)

<a id="tag-dyadic"></a>
`dyadic` (1) — [DiscreteWaveletTransform](domain/spectral.md)

<a id="tag-envelope"></a>
`envelope` (1) — [AnalyticSignal](domain/transforms.md)

<a id="tag-frequency-band"></a>
`frequency-band` (2) — [BandPower](domain/spectral.md), [BandpassFilter](domain/transforms.md)

<a id="tag-frequency-domain"></a>
`frequency-domain` (4) — [Coherence](domain/connectivity.md), [DirectedTransferFunction](domain/connectivity.md), [FourierTransform](domain/transforms.md), [PartialDirectedCoherence](domain/connectivity.md)

<a id="tag-gamma"></a>
`gamma` (1) — [BandPower](domain/spectral.md)

<a id="tag-instantaneous-frequency"></a>
`instantaneous-frequency` (1) — [AnalyticSignal](domain/transforms.md)

<a id="tag-instantaneous-phase"></a>
`instantaneous-phase` (1) — [AnalyticSignal](domain/transforms.md)

<a id="tag-power-spectrum"></a>
`power-spectrum` (2) — [BandPower](domain/spectral.md), [Spectrogram](domain/spectral.md)

<a id="tag-relative-power"></a>
`relative-power` (1) — [Cordance](domain/spectral.md)

<a id="tag-scale-adaptive"></a>
`scale-adaptive` (1) — [ContinuousWaveletTransform](domain/spectral.md)

<a id="tag-scalogram"></a>
`scalogram` (1) — [ContinuousWaveletTransform](domain/spectral.md)

<a id="tag-sub-band"></a>
`sub-band` (1) — [DiscreteWaveletTransform](domain/spectral.md)

<a id="tag-theta"></a>
`theta` (1) — [BandPower](domain/spectral.md)

<a id="tag-time-domain"></a>
`time-domain` (1) — [InverseFourierTransform](domain/transforms.md)

<a id="tag-time-frequency"></a>
`time-frequency` (4) — [AnalyticSignal](domain/transforms.md), [ContinuousWaveletTransform](domain/spectral.md), [EMD](domain/decompositions.md), [Spectrogram](domain/spectral.md)

## Signal property

<a id="tag-algorithmic-complexity"></a>
`algorithmic-complexity` (1) — [LempelZiv](domain/infometrics.md)

<a id="tag-binary"></a>
`binary` (1) — [LempelZiv](domain/infometrics.md)

<a id="tag-data-driven"></a>
`data-driven` (1) — [EMD](domain/decompositions.md)

<a id="tag-dynamical-systems"></a>
`dynamical-systems` (2) — [RecurrenceMatrix](domain/infometrics.md), [SampleEntropy](domain/infometrics.md)

<a id="tag-intrinsic-mode-functions"></a>
`intrinsic-mode-functions` (1) — [EMD](domain/decompositions.md)

<a id="tag-lag"></a>
`lag` (1) — [Autocorrelation](domain/signalstats.md)

<a id="tag-linear"></a>
`linear` (2) — [Correlation](domain/connectivity.md), [Covariance](domain/connectivity.md)

<a id="tag-nonlinear"></a>
`nonlinear` (2) — [EMD](domain/decompositions.md), [MutualInformation](domain/connectivity.md)

<a id="tag-nonlinear-dynamics"></a>
`nonlinear-dynamics` (2) — [Nonreversibility](domain/infometrics.md), [RecurrenceMatrix](domain/infometrics.md)

<a id="tag-nonstationary"></a>
`nonstationary` (1) — [EMD](domain/decompositions.md)

<a id="tag-patterns"></a>
`patterns` (1) — [SVD](domain/decompositions.md)

<a id="tag-predictability"></a>
`predictability` (1) — [SampleEntropy](domain/infometrics.md)

<a id="tag-probability-distribution"></a>
`probability-distribution` (1) — [AmplitudeEntropy](domain/infometrics.md)

<a id="tag-regularity"></a>
`regularity` (1) — [SampleEntropy](domain/infometrics.md)

<a id="tag-self-similarity"></a>
`self-similarity` (2) — [FractalDimension](domain/infometrics.md), [RecurrenceMatrix](domain/infometrics.md)

<a id="tag-signal-complexity"></a>
`signal-complexity` (1) — [LineLength](domain/signalstats.md)

<a id="tag-standard-deviation"></a>
`standard-deviation` (1) — [AmplitudeVariation](domain/signalstats.md)

<a id="tag-state-space"></a>
`state-space` (1) — [RecurrenceMatrix](domain/infometrics.md)

<a id="tag-stationarity"></a>
`stationarity` (1) — [Autocorrelation](domain/signalstats.md)

<a id="tag-temporal-dependence"></a>
`temporal-dependence` (1) — [Autocorrelation](domain/signalstats.md)

<a id="tag-temporal-dynamics"></a>
`temporal-dynamics` (2) — [SlidingWindow](domain/windowing.md), [SlidingWindowReduce](domain/windowing.md)

<a id="tag-time-irreversibility"></a>
`time-irreversibility` (1) — [Nonreversibility](domain/infometrics.md)

<a id="tag-variability"></a>
`variability` (2) — [AmplitudeVariation](domain/signalstats.md), [LineLength](domain/signalstats.md)

## Operation

<a id="tag-aggregation"></a>
`aggregation` (2) — [ConcatAggregate](domain/windowing.md), [MeanAggregate](domain/windowing.md)

<a id="tag-decomposition"></a>
`decomposition` (1) — [DiscreteWaveletTransform](domain/spectral.md)

<a id="tag-denoising"></a>
`denoising` (1) — [DiscreteWaveletTransform](domain/spectral.md)

<a id="tag-dimensionality-reduction"></a>
`dimensionality-reduction` (1) — [SVD](domain/decompositions.md)

<a id="tag-filtering"></a>
`filtering` (1) — [BandpassFilter](domain/transforms.md)

<a id="tag-null-hypothesis"></a>
`null-hypothesis` (1) — [FourierTransformSurrogates](domain/surrogates.md)

<a id="tag-outlier-detection"></a>
`outlier-detection` (1) — [SpikeCount](domain/signalstats.md)

<a id="tag-post-processing"></a>
`post-processing` (1) — [ReciprocalConnectivity](domain/connectivity.md)

<a id="tag-preprocessing"></a>
`preprocessing` (1) — [BandpassFilter](domain/transforms.md)

<a id="tag-reduction"></a>
`reduction` (4) — [Max](domain/signalstats.md), [Mean](domain/signalstats.md), [Min](domain/signalstats.md), [SlidingWindowReduce](domain/windowing.md)

<a id="tag-segmentation"></a>
`segmentation` (2) — [SlidingWindow](domain/windowing.md), [SlidingWindowReduce](domain/windowing.md)

<a id="tag-source-localization"></a>
`source-localization` (1) — [SVD](domain/decompositions.md)

## IO shape

<a id="tag-io-adds-dimension"></a>
`io:adds-dimension` (3) — [BandpassFilter](domain/transforms.md), [DiscreteWaveletTransform](domain/spectral.md), [EMD](domain/decompositions.md)

<a id="tag-io-frequency-output"></a>
`io:frequency-output` (1) — [FourierTransform](domain/transforms.md)

<a id="tag-io-frequency-time-output"></a>
`io:frequency-time-output` (2) — [ContinuousWaveletTransform](domain/spectral.md), [Spectrogram](domain/spectral.md)

<a id="tag-io-iterator"></a>
`io:iterator` (2) — [FourierTransformSurrogates](domain/surrogates.md), [SlidingWindow](domain/windowing.md)

<a id="tag-io-matrix"></a>
`io:matrix` (10) — [Coherence](domain/connectivity.md), [Correlation](domain/connectivity.md), [Covariance](domain/connectivity.md), [EnvelopeCorrelation](domain/connectivity.md), [GrangerCausality](domain/connectivity.md), [MutualInformation](domain/connectivity.md), [PartialCorrelation](domain/connectivity.md), [PhaseLockingValue](domain/connectivity.md), [RecurrenceMatrix](domain/infometrics.md), [SVD](domain/decompositions.md)

<a id="tag-io-matrix-frequency"></a>
`io:matrix-frequency` (2) — [DirectedTransferFunction](domain/connectivity.md), [PartialDirectedCoherence](domain/connectivity.md)

<a id="tag-io-preserves-time"></a>
`io:preserves-time` (4) — [AnalyticSignal](domain/transforms.md), [BandpassFilter](domain/transforms.md), [ConcatAggregate](domain/windowing.md), [SlidingWindowReduce](domain/windowing.md)

<a id="tag-io-scalar"></a>
`io:scalar` (3) — [Max](domain/signalstats.md), [Mean](domain/signalstats.md), [Min](domain/signalstats.md)

<a id="tag-io-scalar-per-channel"></a>
`io:scalar-per-channel` (11) — [AmplitudeEntropy](domain/infometrics.md), [AmplitudeVariation](domain/signalstats.md), [Autocorrelation](domain/signalstats.md), [EpileptogenicityIndex](domain/signalstats.md), [FractalDimension](domain/infometrics.md), [LempelZiv](domain/infometrics.md), [LineLength](domain/signalstats.md), [MeanAggregate](domain/windowing.md), [Nonreversibility](domain/infometrics.md), [SampleEntropy](domain/infometrics.md), [SpikeCount](domain/signalstats.md)

<a id="tag-io-scalar-per-channel-per-band"></a>
`io:scalar-per-channel-per-band` (2) — [BandPower](domain/spectral.md), [Cordance](domain/spectral.md)

<a id="tag-io-time-output"></a>
`io:time-output` (1) — [InverseFourierTransform](domain/transforms.md)

<a id="tag-io-vector"></a>
`io:vector` (1) — [ReciprocalConnectivity](domain/connectivity.md)

## Requirement

<a id="tag-req-asymmetric-matrix"></a>
`req:asymmetric-matrix` (1) — [ReciprocalConnectivity](domain/connectivity.md)

<a id="tag-req-frequency-input"></a>
`req:frequency-input` (1) — [InverseFourierTransform](domain/transforms.md)

<a id="tag-req-multichannel"></a>
`req:multichannel` (10) — [Coherence](domain/connectivity.md), [Correlation](domain/connectivity.md), [Covariance](domain/connectivity.md), [DirectedTransferFunction](domain/connectivity.md), [EnvelopeCorrelation](domain/connectivity.md), [GrangerCausality](domain/connectivity.md), [MutualInformation](domain/connectivity.md), [PartialCorrelation](domain/connectivity.md), [PartialDirectedCoherence](domain/connectivity.md), [PhaseLockingValue](domain/connectivity.md)

<a id="tag-req-sampling-rate"></a>
`req:sampling-rate` (5) — [Coherence](domain/connectivity.md), [DirectedTransferFunction](domain/connectivity.md), [GrangerCausality](domain/connectivity.md), [PartialDirectedCoherence](domain/connectivity.md), [PhaseLockingValue](domain/connectivity.md)
