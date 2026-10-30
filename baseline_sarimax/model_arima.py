"""
Module: model_arima.py
Mục đích: Xây dựng lớp mô hình Baseline SARIMA (Seasonal ARIMA)
          chuẩn hóa cho bài toán dự báo chuỗi thời gian bán lẻ M5 (Walmart).

Nguyên tắc thiết kế (Hướng A — No Fallback):
    - Nếu statsmodels không thể fit/predict → ném SARIMAFitError / SARIMAPredictError.
    - Không bao giờ thay thế bằng phương pháp khác (Seasonal Naive, v.v.).
    - Điều này đảm bảo mọi kết quả metrics đều là của mô hình SARIMA thực sự.

Mô hình:
    SARIMAX(p, d, q)(P, D, Q)_s với s=7 (chu kỳ tuần).
    Khi use_exog=False: đây là SARIMA thuần (không phải SARIMAX).
"""

import warnings
import numpy as np
from typing import Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


# ── Custom Exceptions ──────────────────────────────────────────────────────────

class SARIMANotFittedError(RuntimeError):
    """Ném ra khi predict() được gọi trước khi fit()."""
    pass


class SARIMAFitError(RuntimeError):
    """Ném ra khi statsmodels không thể fit mô hình SARIMA/SARIMAX."""
    pass


class SARIMAPredictError(RuntimeError):
    """Ném ra khi statsmodels không thể tạo dự báo."""
    pass


# ── Model ──────────────────────────────────────────────────────────────────────

class M5ARIMABaseline:
    """
    SARIMA(p, d, q)(P, D, Q)_s baseline cho bài toán dự báo nhu cầu bán lẻ M5.

    Phương trình mô hình:
        Phi_P(B^s) * phi_p(B) * (1-B^s)^D * (1-B)^d * y_t
            = Theta_Q(B^s) * theta_q(B) * epsilon_t

    Hỗ trợ:
      - ARIMA(p, d, q)                        khi seasonal_order=None
      - SARIMA(p, d, q)(P, D, Q)_s            khi use_exog=False (mặc định)
      - SARIMAX(p, d, q)(P, D, Q)_s + X       khi use_exog=True

    Không có fallback: nếu fit/predict thất bại → SARIMAFitError/SARIMAPredictError.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = (1, 1, 1, 7),
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
        name: str = "SARIMA",
    ):
        """
        Args:
            order:               (p, d, q) — bậc phi tuyến AR, sai phân, MA.
            seasonal_order:      (P, D, Q, s) — bậc mùa vụ. None → không có mùa vụ.
            enforce_stationarity: Ràng buộc nghiệm nằm trong vòng tròn đơn vị (AR).
            enforce_invertibility: Ràng buộc nghiệm nằm trong vòng tròn đơn vị (MA).
            name:                Tên nhận dạng để in log.
        """
        self.order                = order
        self.seasonal_order       = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility= enforce_invertibility
        self.name                 = name
        self.model_fitted         = None   # SARIMAXResultsWrapper sau khi fit()

    # ── Fit ────────────────────────────────────────────────────────────────────

    def fit(
        self,
        y_train: np.ndarray,
        exog_train: Optional[np.ndarray] = None,
    ) -> "M5ARIMABaseline":
        """
        Huấn luyện mô hình SARIMA / SARIMAX trên chuỗi thời gian lịch sử.

        Args:
            y_train:    Mảng 1-D doanh số theo ngày (non-negative floats).
            exog_train: Ma trận 2-D biến ngoại sinh tuỳ chọn (shape: [n, k]).

        Returns:
            self

        Raises:
            SARIMAFitError: Nếu statsmodels không thể hội tụ tối ưu hóa MLE.
        """
        y_train = np.asarray(y_train, dtype=float)

        try:
            if self.seasonal_order is not None:
                model = SARIMAX(
                    y_train,
                    exog=exog_train,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=self.enforce_stationarity,
                    enforce_invertibility=self.enforce_invertibility,
                )
            else:
                # ARIMA thuần — không truyền enforce_* (không được hỗ trợ)
                model = ARIMA(
                    y_train,
                    exog=exog_train,
                    order=self.order,
                )

            self.model_fitted = model.fit(method_kwargs={"warn_convergence": False})

        except Exception as e:
            # Không dùng fallback — ném exception rõ ràng
            raise SARIMAFitError(
                f"[{self.name}] statsmodels SARIMA{self.order}x{self.seasonal_order} "
                f"khong the fit: {e}"
            ) from e

        return self

    # ── Predict ────────────────────────────────────────────────────────────────

    def predict(
        self,
        horizon: int = 28,
        exog_future: Optional[np.ndarray] = None,
        return_conf_int: bool = True,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Dự báo H bước thời gian kế tiếp bằng SARIMA.

        Args:
            horizon:         Số ngày dự báo (M5 dùng H=28).
            exog_future:     Biến ngoại sinh 28 ngày tương lai (nếu use_exog=True).
            return_conf_int: Nếu True, trả về khoảng tin cậy 95%.
            alpha:           Mức ý nghĩa thống kê (0.05 = CI 95%).

        Returns:
            (forecast_mean, ci_lower, ci_upper)
            ci_lower / ci_upper là None khi return_conf_int=False.

        Raises:
            SARIMANotFittedError: Nếu fit() chưa được gọi.
            SARIMAPredictError:   Nếu statsmodels không thể tạo dự báo.
        """
        if self.model_fitted is None:
            raise SARIMANotFittedError(
                f"[{self.name}] fit() chua duoc goi. Vui long goi fit() truoc predict()."
            )
        if horizon <= 0:
            raise ValueError(f"horizon phai > 0, nhan duoc: {horizon}")

        try:
            fcast = self.model_fitted.get_forecast(steps=horizon, exog=exog_future)
            preds = np.clip(np.asarray(fcast.predicted_mean, dtype=float), 0.0, None)

            lower_b = upper_b = None
            if return_conf_int:
                # conf_int() có thể trả về DataFrame hoặc ndarray tuỳ phiên bản
                ci_raw  = fcast.conf_int(alpha=alpha)
                ci      = ci_raw.values if hasattr(ci_raw, "values") else np.asarray(ci_raw)
                lower_b = np.clip(ci[:, 0].astype(float), 0.0, None)
                upper_b = np.clip(ci[:, 1].astype(float), 0.0, None)

                # Đảm bảo lower <= pred <= upper (sau khi clip về 0)
                lower_b = np.minimum(lower_b, preds)
                upper_b = np.maximum(upper_b, preds)

            return preds, lower_b, upper_b

        except (SARIMANotFittedError, ValueError):
            raise
        except Exception as e:
            raise SARIMAPredictError(
                f"[{self.name}] statsmodels get_forecast gap loi: {e}"
            ) from e

    # ── Utilities ──────────────────────────────────────────────────────────────

    def is_fitted(self) -> bool:
        """Trả về True nếu mô hình đã được fit() thành công."""
        return self.model_fitted is not None

    def summary(self) -> str:
        """Trả về bảng tóm tắt thống kê của mô hình statsmodels."""
        if self.model_fitted is None:
            return f"{self.name}: chua duoc fit."
        return str(self.model_fitted.summary())

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted() else "not fitted"
        return (
            f"M5ARIMABaseline(order={self.order}, "
            f"seasonal_order={self.seasonal_order}, "
            f"name='{self.name}', {status})"
        )
