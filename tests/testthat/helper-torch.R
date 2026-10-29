skip_if_no_torch <- function() {
  testthat::skip_if_not(torch::torch_is_installed(),
                        "libtorch/lantern not installed")
}
