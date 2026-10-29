# Chạy trong RStudio: Session > Set Working Directory > To Source File Location,
# rồi bấm Source. Script tự bỏ qua bước cài đặt nếu torch đã sẵn sàng.

need <- c("torch", "devtools", "testthat", "roxygen2", "jsonlite",
          "knitr", "rmarkdown")
miss <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
if (length(miss)) install.packages(miss)
if (!torch::torch_is_installed()) {
  torch::install_torch()   # tải libtorch + lantern; sau đó khởi động lại R
  stop("Đã cài libtorch. Hãy Session > Restart R, rồi Source lại file này.")
}

devtools::document()
devtools::load_all()

res <- kan_validate_all(backend = "both")
print(res)
cat("ALL PASS:", attr(res, "all_pass"), "\n")

devtools::test()

# R CMD check trực tiếp (không cần Rtools vì package thuần R)
pkg_dir <- getwd()
old <- setwd(dirname(pkg_dir))
Rbin <- file.path(R.home("bin"), "R")
system2(Rbin, c("CMD", "build", basename(pkg_dir)))
tarball <- tail(sort(Sys.glob("splineKANr_*.tar.gz")), 1)
system2(Rbin, c("CMD", "check", "--no-manual", tarball))
setwd(old)

source("inst/validation/run_validation.R")   # benchmark 5 bài toán x 5 seed
