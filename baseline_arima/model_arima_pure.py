"""
Module: model_arima_pure.py
Mục đích: Mô hình Baseline ARIMA(p, d, q) THUẦN TÚY — không có bất kỳ
          thành phần mùa vụ (Seasonal) hay biến ngoại sinh (Exogenous) nào.

Vị trí trong nghiên cứu:
    Đây là mô hình kinh tế lượng cổ điển đơn giản nhất của họ Box-Jenkins,
    dùng để thiết lập mức sàn hiệu năng (performance floor) nhằm so sánh
    với SARIMA, LightGBM và Temporal Fusion Transformer (TFT).

Cấu hình mặc định ARIMA(7, 1, 1):
    - p=7: Tự hồi quy 7 bước (1 tuần) để bắt quán tính ngắn hạn.
    - d=1: Sai phân bậc 1 để làm chuỗi dừng (stationary).
    - q=1: Trung bình trượt bậc 1 để xử lý phần dư.

Nguyên tắc thiết kế (Hướng A — No Fallback):
    - Nếu statsmodels không thể fit/predict → ném ARIMAFitError / ARIMAPredictError.
    - Không bao giờ thay thế bằng phương pháp khác (Seasonal Naive, v.v.).
    - Điều này đảm bảo kết quả đo lường 100% là của ARIMA, không bị lẫn
      phương pháp khác làm mờ đi ranh giới học thuật.
"""

import warnings
import numpy as np
from typing import Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


# ── Custom Exceptions ──────────────────────────────────────────────────────────

class ARIMANotFittedError(RuntimeError):
    """Ném ra khi predict() được gọi trước khi fit()."""
    pass


class ARIMAFitError(RuntimeError):
    """Ném ra khi statsmodels không thể fit mô hình ARIMA."""
    pass


class ARIMAPredictError(RuntimeError):
    """Ném ra khi statsmodels không thể tạo dự báo."""
    pass


# ── Model ──────────────────────────────────────────────────────────────────────

class PureARIMAModel:
    """
    ARIMA(p, d, q) thuần túy — không có mùa vụ, không có biến ngoại sinh.

    Phương trình mô hình:
        phi_p(B) * (1 - B)^d * y_t = theta_q(B) * epsilon_t

    Trong đó:
        phi_p(B)   = 1 - phi_1*B - ... - phi_p*B^p   (đa thức AR bậc p)
        theta_q(B) = 1 + theta_1*B + ... + theta_q*B^q (đa thức MA bậc q)
        B          = Toán tử trễ: B*y_t = y_{t-1}
        d          = Bậc sai phân để khử xu thế (thường d=1)
        epsilon_t  ~ N(0, sigma^2)  (nhiễu trắng)

    Ràng buộc nghiêm ngặt:
        - Không có Seasonal component  → không phải SARIMA
        - Không có Exogenous variables → không phải ARIMAX
        - Không có fallback nào khác   → nếu fit/predict lỗi, ném exception
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (7, 1, 1),
        name: str = "ARIMA",
    ):
        """
        Args:
            order: (p, d, q) — Bậc AR, sai phân, MA.
                   Mặc định (7, 1, 1):
                     p=7 → AR(7) bắt trễ 7 ngày thay cho thành phần mùa vụ.
                     d=1 → sai phân bậc 1 khử xu thế tuyến tính.
                     q=1 → MA(1) hấp thụ cú sốc ngẫu nhiên ngắn hạn.
            name:  Tên nhận dạng để in log (vd: "ARIMA_CA_1").
        """
        if len(order) != 3:
            raise ValueError(f"order phải có đúng 3 phần tử (p, d, q), nhận được: {order}")
        p, d, q = order
        if any(v < 0 for v in (p, d, q)):
            raise ValueError(f"p, d, q phải >= 0, nhận được: {order}")

        self.order = order
        self.name  = name
        self._model_fitted = None   # ARIMAResultsWrapper sau khi fit()

    # ── Fit ────────────────────────────────────────────────────────────────────

    def fit(self, y_train: np.ndarray) -> "PureARIMAModel":
        """
        Huấn luyện mô hình ARIMA(p, d, q) trên chuỗi thời gian lịch sử.

        Args:
            y_train: Mảng 1-D số thực, doanh số theo ngày (non-negative).
                     Phải có độ dài > max(p, d+1, q) để ước lượng được.

        Returns:
            self (để cho phép method chaining)

        Raises:
            ARIMAFitError: Nếu statsmodels không thể hội tụ tối ưu hóa MLE.
        """
        y_train = np.asarray(y_train, dtype=float)

        min_len = max(self.order[0], self.order[1] + 1, self.order[2]) + 10
        if len(y_train) < min_len:
            raise ARIMAFitError(
                f"[{self.name}] Chuoi huan luyen qua ngan: "
                f"can it nhat {min_len} quan sat, chi co {len(y_train)}."
            )

        try:
            model = ARIMA(y_train, order=self.order)
            self._model_fitted = model.fit(method_kwargs={"warn_convergence": False})
        except Exception as e:
            # Không dùng fallback — ném exception rõ ràng để pipeline xử lý
            raise ARIMAFitError(
                f"[{self.name}] statsmodels ARIMA{self.order} khong the fit: {e}"
            ) from e

        return self

    # ── Predict ────────────────────────────────────────────────────────────────

    def predict(
        self,
        horizon: int = 28,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Dự báo H bước thời gian kế tiếp bằng ARIMA.

        Args:
            horizon: Số ngày dự báo (M5 dùng H=28). Phải > 0.
            alpha:   Mức ý nghĩa thống kê cho CI (0.05 = khoảng tin cậy 95%).

        Returns:
            Tuple (forecast_mean, ci_lower, ci_upper) — tất cả là ndarray 1-D
            có độ dài = horizon. Tất cả giá trị được clip về >= 0 (doanh số
            không thể âm).

        Raises:
            ARIMANotFittedError: Nếu fit() chưa được gọi.
            ARIMAPredictError:   Nếu statsmodels không thể tạo dự báo.
        """
        if self._model_fitted is None:
            raise ARIMANotFittedError(
                f"[{self.name}] fit() chua duoc goi. Vui long goi fit() truoc predict()."
            )
        if horizon <= 0:
            raise ValueError(f"horizon phai > 0, nhan duoc: {horizon}")

        try:
            fcast   = self._model_fitted.get_forecast(steps=horizon)
            preds   = np.clip(np.asarray(fcast.predicted_mean, dtype=float), 0.0, None)

            # conf_int() có thể trả về DataFrame hoặc ndarray tuỳ phiên bản
            ci_raw  = fcast.conf_int(alpha=alpha)
            ci      = ci_raw.values if hasattr(ci_raw, "values") else np.asarray(ci_raw)
            lower_b = np.clip(ci[:, 0].astype(float), 0.0, None)
            upper_b = np.clip(ci[:, 1].astype(float), 0.0, None)

            # Đảm bảo lower <= pred <= upper (statsmodels đôi khi bị flip do clip)
            lower_b = np.minimum(lower_b, preds)
            upper_b = np.maximum(upper_b, preds)

            return preds, lower_b, upper_b

        except (ARIMANotFittedError, ValueError):
            raise   # re-raise các exception đã xử lý, không bọc thêm
        except Exception as e:
            raise ARIMAPredictError(
                f"[{self.name}] statsmodels get_forecast gap loi: {e}"
            ) from e

    # ── Utilities ──────────────────────────────────────────────────────────────

    def is_fitted(self) -> bool:
        """Trả về True nếu mô hình đã được fit() thành công."""
        return self._model_fitted is not None

    def summary(self) -> str:
        """Trả về bảng tóm tắt thống kê của mô hình statsmodels."""
        if self._model_fitted is None:
            return f"{self.name}: chua duoc fit."
        return str(self._model_fitted.summary())

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted() else "not fitted"
        return f"PureARIMAModel(order={self.order}, name='{self.name}', {status})"
