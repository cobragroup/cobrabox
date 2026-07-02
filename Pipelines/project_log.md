# Project Log

Newest entries on top.

---

## 2026-07-02

* The project was moved to a new branch of *cobrabox*: "Pipeline_dev". A new folder was created to gather all the tutorial and example notebooks [https://github.com/cobragroup/cobrabox/tree/Pipelines_dev/Pipelines](https://github.com/cobragroup/cobrabox/tree/Pipelines_dev/Pipelines).
* A clean-up of the folder was done to conserve only the needed files.
* A README.md file was added, to summarise the notebooks we will offer.
* Decoration of the notebooks started.
* Reshaping of the content and flows of the three original notebooks was discussed (Christos & Gorka). The new organization was defined in a new file of the shared google drive, see “*COBRABOX_Coordination > WorkFlows > PipelineLayout.docx*”   



## 2026-05-18

### Resolved: notch filter order (before vs. after bipolar montage)

The notch filter is applied to raw unipolar signals before computing bipolar differences.
Raised as a question because the order could in principle matter.

**Decision:** order is irrelevant. Both the notch filter and bipolar subtraction are linear
operations, so `notch(a − b) = notch(a) − notch(b)`. The results are bit-identical either way.

### Resolved: averaging order (segments vs. frequency)

The pipeline averages PDC over segments first, then over frequencies within each band.
Raised as a question because different orderings could give different results.

**Decision:** order is irrelevant for two independent reasons:

1. Arithmetic means commute — `mean_freq(mean_seg(PDC)) = mean_seg(mean_freq(PDC))` exactly.
2. RC itself is linear in the PDC matrix (it is just `in_strength − out_strength`, a sum of
   matrix entries), so averaging PDC before computing RC gives the same result as computing RC
   per segment and averaging afterwards.

The only ordering choice that would matter is if a nonlinear step (e.g. per-segment
normalization, thresholding) were introduced between averaging steps. There is none here.

### Bug fix: `band_matrices` in multi-subject loop

`rc_per_band` dict comprehension iterated `band_matrices.items()` instead of `BANDS.keys()`.
`band_matrices` is only computed in the exploratory sub-01 cell and is not updated inside the
loop. The variable `matrix` from the iteration was never used in the value expression, so
results happened to be correct as long as the kernel was not restarted before running the loop.
Fixed to iterate `BANDS.keys()` directly.
