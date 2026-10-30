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
"""

import warnings
import numpy as np
from typing import Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


class PureARIMAModel:
    """
    ARIMA(p, d, q) thuần túy — không có mùa vụ, không có biến ngoại sinh.

    Phương trình mô hình:
        phi(B) * (1 - B)^d * y_t = theta(B) * epsilon_t

    Trong đó:
        phi(B)   = 1 - phi_1*B - phi_2*B^2 - ... - phi_p*B^p   (AR polynomial)
        theta(B) = 1 + theta_1*B + ... + theta_q*B^q            (MA polynomial)
        B        = Toán tử trễ (Backshift operator): B*y_t = y_{t-1}
        d        = Bậc sai phân để khử xu thế (thường d=1)
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (7, 1, 1),
        name: str = "ARIMA",
    ):
        """
        Args:
            order: (p, d, q) — Bậc AR, sai phân, MA.
                   Mặc định (7, 1, 1): AR bậc 7 bắt trễ 1 tuần.
            name:  Tên mô hình để in log.
        """
        self.order = order
        self.name  = name
        self._model_fitted  = None
        self._y_train: Optional[np.ndarray] = None

    # ── Fit ────────────────────────────────────────────────────────────────────

    def fit(self, y_train: np.ndarray) -> "PureARIMAModel":
        """
        Huấn luyện mô hình ARIMA trên chuỗi thời gian lịch sử.

        Args:
            y_train: Mảng 1-D doanh số theo ngày (non-negative floats).
        """
        y_train = np.asarray(y_train, dtype=float)
        self._y_train = y_train.copy()

        try:
            model = ARIMA(y_train, order=self.order)
            self._model_fitted = model.fit(disp=False, maxiter=300)
        except Exception as e:
            warnings.warn(
                f"[{self.name}] fit() order={self.order} gap loi: {e}. "
                "Thu fallback ARIMA(1,1,1)..."
            )
            try:
                self._model_fitted = ARIMA(y_train, order=(1, 1, 1)).fit(disp=False)
            except Exception:
                self._model_fitted = None  # hoàn toàn thất bại → dùng naive

        return self

    # ── Predict ────────────────────────────────────────────────────────────────

    def predict(
        self,
        horizon: int = 28,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Dự báo H bước thời gian kế tiếp.

        Args:
            horizon: Số ngày dự báo (M5 dùng H=28).
            alpha:   Mức ý nghĩa thống kê cho CI (0.05 = CI 95%).

        Returns:
            (forecast_mean, ci_lower_95, ci_upper_95)
            ci_lower/ci_upper là None khi mô hình bị fallback.
        """
        if self._model_fitted is not None:
            try:
                fcast = self._model_fitted.get_forecast(steps=horizon)
                preds = np.clip(np.asarray(fcast.predicted_mean), 0, None)

                # conf_int() trả về DataFrame → dùng .values để lấy ndarray
                ci      = fcast.conf_int(alpha=alpha).values
                lower_b = np.clip(ci[:, 0], 0, None)
                upper_b = np.clip(ci[:, 1], 0, None)

                return preds, lower_b, upper_b

            except Exception as e:
                warnings.warn(f"[{self.name}] predict() gap loi: {e}. Dung naive fallback.")

        # Seasonal Naive fallback (tránh bias 1 điểm)
        preds = self._seasonal_naive(horizon)
        return preds, None, None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _seasonal_naive(self, horizon: int, period: int = 7) -> np.ndarray:
        """
        Fallback: trung bình 4 tuần lịch sử cùng ngày trong tuần.
        Chỉ được kích hoạt khi statsmodels ARIMA fit/predict thất bại.
        """
        if self._y_train is None or len(self._y_train) == 0:
            return np.zeros(horizon)

        n    = len(self._y_train)
        preds = np.zeros(horizon)
        for h in range(horizon):
            idx = [n - period * k + h for k in range(4, 0, -1)]
            vals = [self._y_train[i] for i in idx if 0 <= i < n]
            preds[h] = float(np.mean(vals)) if vals else 0.0
        return np.clip(preds, 0, None)

    def summary(self) -> str:
        """Trả về tóm tắt thống kê mô hình statsmodels."""
        if self._model_fitted is None:
            return f"{self.name}: model not fitted."
        return str(self._model_fitted.summary())

