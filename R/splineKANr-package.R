#' splineKANr: B-spline Kolmogorov-Arnold Networks in native R torch
#'
#' Main entry points:
#' * [kan_fit()] / [predict.kan_model()] / [kan_metrics()]: training and
#'   evaluation.
#' * [kan_update_grid()], [kan_refine()], [kan_reg_control()],
#'   [kan_prune()]: KAN-specific tools.
#' * [plot.kan_model()], [kan_plot_network()], [kan_partial_dependence()]:
#'   interpretation.
#' * [kan_save()] / [kan_load()]: persistence.
#' * [kan_validate_all()], [kan_benchmark()]: numerical validation.
#'
#' Every model is a plain R object; [kan_forward_r()] and
#' [kan_gradient_r()] evaluate it without libtorch and serve as the
#' reference implementation for the torch backend.
#'
#' @references Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J.,
#'   Soljacic, M., Hou, T. Y. and Tegmark, M. (2024). KAN: Kolmogorov-Arnold
#'   Networks. \doi{10.48550/arXiv.2404.19756}
#' @keywords internal
"_PACKAGE"
