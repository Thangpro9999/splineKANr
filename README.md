# splineKANr

B-spline Kolmogorov-Arnold Networks (KAN) in native R `torch`, with a
dependency-free base R reference implementation used as a numerical oracle.

```r
# install.packages(c("torch", "devtools")); torch::install_torch()
devtools::install()          # from the package folder
library(splineKANr)

d   <- kan_synthetic("interaction", n = 500, seed = 1)
fit <- kan_fit(d$x, d$y, hidden = 4, epochs = 200, batch_size = 64,
               grid_update_every = 10, seed = 1)
predict(fit, d$x[1:5, ])
plot(fit, layer = 1, data = d$x)
kan_validate_all()           # basis, derivative, gradient checks
```

Status: see `NEWS.md`, `inst/validation/` and the verification report
shipped with each release. pykan (Liu et al., 2024) is used only to produce
reference numbers; the package does not depend on Python.

## License and citation

splineKANr is copyright 2026 Thang Le-Dinh and is released under the
Apache License 2.0 (`LICENSE.md`). Redistributions and derivative works must
keep the attribution notices in `inst/NOTICE` (Apache-2.0, Section 4(d)).

If you use splineKANr in academic work, please cite it:

```r
citation("splineKANr")
```

GitHub also reads `CITATION.cff` ("Cite this repository").
