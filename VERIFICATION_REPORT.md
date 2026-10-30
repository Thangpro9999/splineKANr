# splineKANr 0.5.0 — Báo cáo kiểm định (2026-10-29)

## Bản 0.5.0: bốn việc tiếp theo (không viết lại lõi B-spline)

Mọi số liệu dưới đây là kết quả chạy thật trong sandbox Linux (R 4.5.3, nhánh
base R; libtorch không có trong sandbox). Lệnh tái lập: `inst/REPRODUCE.md`.
Không có ngưỡng kiểm định nào được nới để test "qua".

### 1. Đối chứng pykan 0.2.8: tương đương số học và tương đương thuật toán

**Tương đương số học** (cùng knots, hệ số, trọng số, cấu trúc mạng, đầu vào):

| Mức | Kết quả |
|---|---|
| KANLayer: basis, forward, gradient, cập nhật grid (0.3.x) | sai lệch < 1e-10 |
| MultKAN 2–3 lớp: forward, loss, gradient mọi lớp, 5 bước Adam (0.4.0) | < 1e-10 |
| `MultKAN.fit(opt = "Adam")` 300 bước, không cập nhật grid | loss từng bước lệch tương đối ≤ 1.4e-14; tham số cuối ≤ 3e-15 |
| 200 bước, có lịch cập nhật grid của pykan (10 lần), `grid_ridge = 0` | loss ≤ 1.3e-13; tham số ≤ 2e-13 |
| như trên, mặc định `grid_ridge = 1e-8` | loss lệch 6e-6 (sin), 3e-4 (additive); tham số ≤ 2e-3 |

Khác biệt ở dòng cuối đã được truy nguyên: chỉ do bước khớp lại hệ số sau
khi cập nhật grid dùng ridge 1e-8 (pykan dùng `lstsq` thuần). Kiểm chứng:
ridge 1e-8 → 1e-12 → 0 cho sai lệch 6e-5 → 6e-9 → 3e-13. Đã thêm tham số
`kan_fit(grid_ridge = )` (mặc định giữ 1e-8, hành vi cũ không đổi). Test mới:
`tests/testthat/test-pykan-training.R` (12 kiểm định).

**Tương đương thuật toán** (mỗi gói tự khởi tạo ngẫu nhiên, cùng dữ liệu,
cùng quy trình: 500 bước Adam lr 0.01, lịch grid của pykan; 10 seed, nhiễu
sd 0.1; so sánh phân phối RMSE ngoài mẫu, ghép cặp theo seed):

| Bài toán | splineKANr | pykan | Chênh lệch TB (KTC 95%) | Wilcoxon p |
|---|---|---|---|---|
| sin [1,1] | 0.1005 (0.0013) | 0.1004 (0.0012) | 0.0001 (−0.0000; 0.0002) | 0.43 |
| interaction [2,4,1] | 0.1087 (0.0022) | 0.1100 (0.0028) | −0.0013 (−0.0028; 0.0002) | 0.06 |
| additive [2,5,1] | 0.1189 (0.0223) | 0.1155 (0.0076) | 0.0035 (−0.0099; 0.0169) | 0.92 |

Không có khác biệt có ý nghĩa thống kê. Độ phân tán lớn hơn của splineKANr ở
bài additive do một seed (RMSE 0.181) hội tụ về nghiệm kém. Khác biệt thuật
toán đã biết và ghi trong tài liệu: ridge khi khớp lại hệ số; cách khởi tạo hệ
số (nhiễu trực tiếp trên hệ số, pykan khớp nhiễu tại điểm lưới); ngoại suy
mặc định `"linear"` (pykan: `"none"`); LBFGS của torch khác LBFGS riêng của
pykan (so sánh LBFGS cần nhánh torch, chưa chạy).

### 2. Benchmark ngoài mẫu (6 bài toán × 10 seed, n_train 500, n_test 1000)

Cùng dữ liệu và tập kiểm tra cho mọi phương pháp; phương pháp có tham số
chọn cấu hình trên cùng một hold-out bên trong (KAN: 4 cấu hình, ANN: 9).
RMSE so với hàm thật (trung bình 10 seed):

| Bài toán | B-spline KAN | Fourier KAN | ANN (nnet) | GAM | Random forest | OLS |
|---|---|---|---|---|---|---|
| sin | 0.0052 | 0.0209 | 0.0043 | **0.0011** | 0.0072 | 0.447 |
| poly | **0.0019** | 0.0070 | 0.0036 | 0.0034 | 0.0037 | 0.153 |
| noisy_sin | 0.0293 | 0.0696 | **0.0234** | 0.0255 | 0.1203 | 0.447 |
| interaction | **0.0058** | 0.0253 | 0.0070 | 0.3343 | 0.1094 | 0.334 |
| additive | 0.0109 | 0.0463 | 0.0143 | **0.0070** | 0.1085 | 0.625 |
| Friedman 1 (p = 10) | 1.207 | 1.569 | **0.570** | 1.394 | 2.044 | 2.441 |

B-spline KAN tốt nhất ở poly và interaction, gần tốt nhất ở bài trơn/cộng
tính (GAM khớp đúng cấu trúc nên tốt hơn), luôn tốt hơn rõ Fourier KAN. Ở
Friedman 1, ANN chính xác gấp đôi và KAN dao động mạnh giữa các seed (sd
0.35): một lớp ẩn rộng 3 không đủ linh hoạt. KAN (nhánh base R) cũng chậm
nhất: 3–18 giây mỗi lần fit. Fourier KAN là bản cài đặt base R chỉ để đối
chứng (gradient kiểm bằng sai phân hữu hạn, sai lệch 1e-8), không thuộc API.

### 3. Ứng dụng quản trị: PLS-SEM → latent scores → KAN

Dữ liệu công khai: ECSI `seminr::mobi` (n = 250), `plspm::satisfaction`
(n = 250), `seminr::corp_rep_data` (n = 344, cấu trúc formative). Dữ liệu
EDU và manuscript không được dùng theo yêu cầu, đã xoá khỏi sandbox.

PLS-PM base R (`inst/applications/pls_pm.R`, mode A/B, path weighting) khớp
`plspm::plspm()`: hệ số đường dẫn < 1e-7, loadings < 2e-7, tương quan điểm số
= 1 trên cả ba bộ. Với `corp_rep_data` tái lập đúng kết quả trong sách Hair
và cs. (2022): QUAL→COMP 0.430, LIKE→CUSA 0.436, CUSA→CUSL 0.505, R² COMP
0.631, LIKE 0.558, CUSA 0.292, CUSL 0.562.

CV 10 × 5-fold; trọng số PLS ước lượng lại trong từng fold huấn luyện, điểm
số fold kiểm tra tính bằng trọng số/trung bình/độ lệch chuẩn của fold huấn
luyện (không rò rỉ). RMSE của điểm chuẩn hoá (trung bình 10 lần lặp):

| Dữ liệu, biến mục tiêu | PLS-SEM | B-spline KAN | GAM | ANN | RF | Fourier KAN |
|---|---|---|---|---|---|---|
| mobi, hài lòng | **0.584** | 0.616 | 0.592 | 0.598 | 0.623 | 0.657 |
| mobi, trung thành | **0.753** | 0.776 | 0.770 | 0.792 | 0.753 | 0.832 |
| satisfaction, hài lòng | **0.567** | 0.601 | 0.599 | 0.590 | 0.587 | 0.630 |
| satisfaction, trung thành | 0.723 | 0.722 | **0.710** | 0.719 | 0.718 | 0.767 |
| corp_rep, hài lòng | **0.852** | 0.864 | 0.860 | 0.869 | 0.899 | 0.898 |
| corp_rep, trung thành | **0.667** | 0.682 | 0.679 | 0.678 | 0.714 | 0.720 |

Kết luận trung thực: trên các bộ này mô hình cấu trúc tuyến tính dự báo
bằng hoặc tốt hơn mọi phương pháp linh hoạt; KAN ngang GAM/ANN. Với 250–350
người trả lời, không có phi tuyến đủ mạnh để bù số tham số tăng thêm. Giá
trị của KAN ở đây là công cụ chẩn đoán: đường partial dependence cho phép
kiểm tra trực quan giả định tuyến tính, và CV cho thấy nới giả định đó không
cải thiện dự báo.

### 4. Chuẩn bị phát hành

- Vignette học thuật mới `vignettes/methods.Rmd` (mô hình, ước lượng, kiểm
  định số học/thuật toán, benchmark, ứng dụng PLS-SEM với dữ liệu mô phỏng
  chạy được trên CRAN) + `references.bib`.
- `inst/REPRODUCE.md`: lệnh, seed, phiên bản phần mềm cho mọi kết quả.
- Giấy phép: từ 0.5.2 là giấy phép học thuật, phi thương mại (`file LICENSE`; 0.5.0–0.5.1 là MIT); phụ thuộc torch (MIT), base R. Không phân phối lại dữ
  liệu bên thứ ba (seminr, plspm là GPL-3); fixture pykan chỉ là số liệu đầu
  ra (pykan MIT), không chép mã.
- API: 37 hàm export, không đổi tên; thêm `kan_fit(grid_ridge = )`; sửa lỗi
  `plot.kan_pd()` khi truyền `xlab`/`ylab` (phát hiện khi chạy ứng dụng, có
  test).
- Test: 223 pass, 0 fail, 12 skip (cần libtorch). `R CMD check --as-cran`:
  0 lỗi, 0 cảnh báo, 3 NOTE (new submission; DOI và thời gian không kiểm tra
  được vì sandbox không có internet; thiếu V8).
- Windows 11 / R 4.6.0 / torch 0.17.0 (máy người dùng, 2026-10-29 18:54 ICT):
  `kan_validate_all(backend = "both")` all pass (torch autograd vs giải tích
  ≤ 3.3e-14); `devtools::test()` 251 pass, 0 fail, 0 skip (gồm 28 kiểm định
  torch-backend); `R CMD check --no-manual` Status: OK; benchmark torch 5
  seed: sin RMSE 2.65e-3, interaction 3.97e-3, noisy_sin 0.2024 (≈ nhiễu 0.2).
- Còn chờ: kết quả mac-builder, win-builder.

---

# Báo cáo các bản trước

## Phase 6: đối chiếu pykan cho mạng nhiều lớp (bản 0.4.0)

So với `MultKAN` của pykan 0.2.8 (float64), trên 3 kiến trúc 2-3-1, 3-4-2-1
và 2-5-3-2, splineKANr dùng `extrapolate = "none"`, `bias = FALSE`:

| Hạng mục | Sai số tối đa | Ngưỡng | Kết quả |
|---|---|---|---|
| Forward toàn mạng | < 1e-10 | 1e-10 | ĐẠT |
| Loss | rel < 1e-12 | 1e-12 | ĐẠT |
| Gradient coef / scale_base / scale_sp mọi lớp | < 1e-10 | 1e-10 | ĐẠT |
| Quỹ đạo 5 bước Adam (loss và hệ số sau cùng) | < 1e-10 / 1e-9 | — | ĐẠT |
| Knot sau grid update toàn mạng, lớp 1 | 2.2e-16 | 1e-12 | ĐẠT |
| Knot sau grid update, lớp sâu hơn | 5.7e-08 … 3.3e-06 | 1e-5 | ĐẠT |

Sai khác nhỏ ở lớp sâu là do cách giải lại hệ số: pykan dùng least squares,
splineKANr dùng ridge 1e-8, nên đầu vào của lớp sau lệch rất nhẹ.
49 kiểm tra trong `tests/testthat/test-pykan-multilayer.R`; script sinh dữ liệu:
`inst/validation/pykan_multilayer_reference.py`.

Kiểm tra lại 0.4.0 trên sandbox Linux: testthat 208 pass / 0 fail (12 test
torch skip vì sandbox không có libtorch); `R CMD check --as-cran`: 0 ERROR,
0 WARNING, 2 NOTE do môi trường.


## Cập nhật: nhánh torch đã kiểm định trên máy người dùng

Windows 11 x64, R 4.6.0, torch 0.17.0 (libtorch + lantern), bản 0.3.1:

- `kan_validate_all(backend = "both")`: **all_pass = TRUE**. Basis torch vs
  splineDesign 3.3e-16; đạo hàm autograd torch vs splineDesign 7.4e-14;
  gradient autograd torch vs giải tích 3.3e-14 (3 kiến trúc × 3 chính sách).
- `devtools::test()`: **FAIL 0 | WARN 0 | SKIP 0 | PASS 187** (gồm 28 test
  torch: forward, autograd, regularization, huấn luyện Adam/L-BFGS, grid
  update, tái lập seed, lưu/nạp, quỹ đạo Adam torch = base R).
- `R CMD check --no-manual` (0.3.1): Status: 1 NOTE (thiếu import `predict`,
  `str`) — đã sửa trong 0.3.2. Sandbox Linux kiểm tra lại 0.3.2:
  0 ERROR, 0 WARNING, 2 NOTE do môi trường.
- Bản 0.3.0 trên Windows có 2 lỗi đã sửa ở 0.3.1: hằng số regularization bị
  ép float32 (lệch 1.5e-7); NaN một lần ở đạo hàm autograd G=10, k=3 (viết
  lại phép tính, thêm test lặp 5 lần — đạt).

### Bản 0.3.2 trên Windows (chạy lại toàn bộ, 2026-10-29 14:00 ICT)

- `kan_validate_all(backend = "both")`: all_pass = TRUE.
- `devtools::test()`: **FAIL 0 | WARN 0 | SKIP 0 | PASS 187**.
- `R CMD check --no-manual`: **Status: OK** (0 ERROR, 0 WARNING, 0 NOTE).

### Benchmark hội tụ nhánh torch (Windows, 5 seed, n_train 400 / n_test 1000)

Adam lr = 0.02, batch 64, ≤ 300 epoch, patience 40. Mean ± sd qua 5 seed:

| Bài toán | Kiến trúc | RMSE test | R² test | RMSE so với hàm thật |
|---|---|---|---|---|
| linear | [1,1] | 1.2e-05 ± 3.1e-06 | 1.0000 | 1.2e-05 |
| sin | [1,1] | 2.65e-03 ± 6.9e-05 | 0.99999 | 2.65e-03 |
| poly | [1,1] | 5.7e-06 ± 6.3e-06 | 1.0000 | 5.7e-06 |
| interaction x1·x2 | [2,4,1] | 3.97e-03 ± 9.2e-04 | 0.99985 | 3.97e-03 |
| noisy_sin (sd 0.2) | [1,1] | 0.2024 ± 0.0023 | 0.9229 | 0.0243 ± 0.0049 |

Trên dữ liệu nhiễu, RMSE test ≈ độ lệch chuẩn nhiễu (0.2) nên mô hình không
overfit; sai số so với hàm thật chỉ 0.024. Thời gian mỗi lần fit 4–34 giây
trên CPU.

Phần bên dưới là báo cáo gốc từ sandbox (trước khi có torch).


Mọi kết quả dưới đây là **kết quả chạy thật** trong sandbox Linux
(Ubuntu 24.04, R 4.5.3 từ conda-forge, torch R 0.17.0, testthat 3,
roxygen2 8.1.0). Log gốc nằm trong thư mục `logs/` đi kèm.

## Giới hạn quan trọng

Sandbox không tải được libtorch/lantern (mạng chặn `torch-cdn.mlverse.org`,
`download.pytorch.org`). Gói `torch` cài được nhưng `torch_is_installed()` =
FALSE, nên **nhánh torch (autograd, huấn luyện bằng torch) chưa được chạy**.
11 test trong `tests/testthat/test-torch-backend.R` bị skip, không phải pass.
Chạy `RUN_IN_RSTUDIO.R` trên máy có libtorch để hoàn tất.

Phần đã chạy dùng bản tham chiếu base R (`backend = "r"`): cùng mô hình,
cùng tham số, đạo hàm giải tích viết tay. Bản này đóng vai trò "oracle":
các test torch so sánh torch với nó.

## Bảng nghiệm thu

| Tiêu chí | Ngưỡng | Kết quả đo | Trạng thái |
|---|---|---|---|
| Basis vs `splines::splineDesign` (miền trong, G=3/5/10, k=1..3) | < 1e-6 | max 3.3e-16 | ĐẠT (R) |
| Partition of unity trên [a, b] | < 1e-6 | max 4.4e-16 | ĐẠT (R) |
| Đạo hàm basis vs splineDesign(derivs=1) | rel < 1e-4 | max 4.4e-14 | ĐẠT (R) |
| Đạo hàm basis vs sai phân hữu hạn | rel < 1e-4 | max 2.1e-6 | ĐẠT (R) |
| Gradient tham số (giải tích) vs sai phân hữu hạn, 3 kiến trúc × 3 chính sách ngoại suy | rel < 1e-3 | max 5.1e-5 | ĐẠT (R) |
| Gradient theo đầu vào vs sai phân hữu hạn | rel < 1e-4 | đạt | ĐẠT (R) |
| Đối chiếu pykan 0.2.8: forward, postacts, loss, grad coef/scale_base/scale_sp, knot sau update_grid | < 1e-10 | đạt (21 kiểm tra) | ĐẠT (R) |
| Hội tụ đa seed (5 bài toán × 3 seed) | xem dưới | đạt ngưỡng | ĐẠT (R) |
| Lưu/nạp: dự báo trùng | < 1e-6 | 0 (bit-identical) | ĐẠT (R) |
| Tái lập theo seed | trùng khớp | `identical()` TRUE | ĐẠT (R) |
| Torch basis / autograd / forward vs base R | 1e-10..1e-12 | đạt (Windows) | ĐẠT |
| Huấn luyện torch, L-BFGS, regularization | — | đạt (Windows) | ĐẠT |
| `R CMD check` | 0 ERROR/WARNING | Windows: Status OK; Linux --as-cran: 2 NOTE môi trường | ĐẠT |
| Đa nền tảng | — | Linux (sandbox) + Windows 11 đạt; macOS chưa chạy | MỘT PHẦN |

2 NOTE của R CMD check đều do môi trường: "unable to verify current time"
(không có mạng) và "package 'V8' unavailable" (bỏ qua kiểm tra hiển thị
công thức trong HTML).

## Benchmark hội tụ (base R, Adam lr=0.02, batch 64, ≤300 epoch, patience 40)

n_train = 400, n_test = 500 (tập kiểm tra độc lập), seed 1..3, mean ± sd:

| Bài toán | Kiến trúc | RMSE test | R² test | RMSE so với hàm thật | Ngưỡng |
|---|---|---|---|---|---|
| linear `2x - 0.5` | [1,1] | 0.0002 ± 0.0002 | 1.0000 | 0.0002 | < 0.02 |
| sin `sin(pi x)` | [1,1] | 0.0025 ± 0.0002 | 1.0000 | 0.0025 | < 0.03 |
| poly `x^3 - 0.5x` | [1,1] | 0.0001 ± 0.0001 | 1.0000 | 0.0001 | < 0.02 |
| interaction `x1 * x2` | [2,4,1] | 0.0048 ± 0.0030 | 0.9997 | 0.0048 | < 0.05 |
| noisy_sin (sd nhiễu 0.2) | [1,1] | 0.2103 ± 0.0049 | 0.9164 | 0.0417 | < 0.08 |

Với dữ liệu nhiễu, RMSE test ≈ độ lệch chuẩn nhiễu (0.2), tức mô hình không
overfit nhiễu; sai số so với hàm thật là 0.04.

## Bốn lỗi kỹ thuật ưu tiên trong lộ trình — cách xử lý

1. **`torch_clamp()` gây gradient bằng 0 ngoài miền**: thay bằng tham số
   `extrapolate`: `"linear"` (mặc định, khai triển Taylor bậc 1 tại biên,
   liên tục C1, gradient khác 0), `"none"` (giống pykan), `"clamp"` (giữ để
   so sánh). Biên phải xử lý như giới hạn trái (`right_closed`). Có test.
2. **Kiểm định đạo hàm**: so sánh 3 chiều giải tích / sai phân hữu hạn /
   splineDesign (đã chạy) và autograd torch (đã viết, chưa chạy). Điểm kiểm
   tra tránh knot.
3. **Ổn định mạng nhiều lớp**: mỗi đầu vào của mỗi lớp có lưới riêng; có
   adaptive grid theo quantile (`kan_update_grid`, khớp knot pykan tới 1e-10),
   `kan_activation_check()` báo tỷ lệ kích hoạt ra ngoài miền sau huấn luyện.
4. **Lưu mô hình**: `kan_save()/kan_load()` lưu cấu trúc, hệ số, knot, mask,
   tham số chuẩn hóa (chỉ ước lượng trên tập train), lịch sử và metadata
   dưới dạng dữ liệu R thuần (không có con trỏ torch).

## Việc còn lại

- Chạy `RUN_IN_RSTUDIO.R` trên máy có libtorch (hoặc mở mạng cho sandbox) để
  xác nhận 11 test torch và benchmark torch.
- Chạy CI GitHub Actions (Windows, macOS, Linux).
- Đối chiếu pykan mở rộng: mạng nhiều lớp MultKAN, cùng quá trình huấn luyện.
- Hồ sơ CRAN: `cran-comments.md`, kiểm tra trên win-builder/R-hub.
