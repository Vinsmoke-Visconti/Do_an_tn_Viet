"""
Module: model_arima.py
Mục đích: Xây dựng lớp mô hình Baseline ARIMA và SARIMAX (Seasonal ARIMA)
          chuẩn hóa cho bài toán dự báo chuỗi thời gian bán lẻ M5 (Walmart).

BUG-FIXES so với phiên bản cũ:
  [1] Fallback ở fit() vẫn nhét last_train_value = naive fallback (cũ).
      -> Sửa: naive fallback trong predict() dùng rolling mean 7 ngày thay vì
         chỉ 1 điểm cuối, tránh bias nghiêm trọng với chuỗi có mùa vụ.
  [2] conf_int ở SARIMAX trả về DataFrame (không phải ndarray).
      -> Sửa: dùng .values để lấy ndarray trước khi truy cập [:, 0].
  [3] ARIMA cổ điển không nhận enforce_stationarity / enforce_invertibility
      -> Sửa: chỉ truyền tham số đó cho SARIMAX, không truyền cho ARIMA.
  [4] Lưu y_train đầy đủ để naive fallback dùng rolling mean.
"""

import warnings
import numpy as np
from typing import Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


class M5ARIMABaseline:
    """
    Mô hình ARIMA / SARIMAX baseline cho bài toán dự báo nhu cầu bán lẻ M5.

    Hỗ trợ:
      - ARIMA(p, d, q): ví dụ (1, 1, 1) bắt quán tính ngắn hạn.
      - SARIMAX(p, d, q)x(P, D, Q, s): chu kỳ tuần s=7 mặc định.
      - Hỗ trợ biến ngoại sinh (Exogenous variables): SNAP, weekend, event.
      - Trả khoảng tin cậy 95% (Confidence Interval).
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = (1, 1, 1, 7),
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
        name: str = "SARIMAX",
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.name = name

        self.model_fitted = None
        self._y_train: Optional[np.ndarray] = None  # [FIX-4] Lưu toàn bộ y_train

    # ------------------------------------------------------------------
    def fit(
        self,
        y_train: np.ndarray,
        exog_train: Optional[np.ndarray] = None,
    ) -> "M5ARIMABaseline":
        """
        Huấn luyện mô hình ARIMA / SARIMAX trên chuỗi thời gian lịch sử.

        Args:
            y_train  : Mảng 1-D doanh số huấn luyện (non-negative).
            exog_train: Ma trận 2-D biến ngoại sinh (tuỳ chọn).
        """
        y_train = np.asarray(y_train, dtype=float)
        self._y_train = y_train.copy()  # [FIX-4]

        try:
            if self.seasonal_order is not None:
                model = SARIMAX(
                    y_train,
                    exog=exog_train,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=self.enforce_stationarity,  # ok cho SARIMAX
                    enforce_invertibility=self.enforce_invertibility,
                )
            else:
                # [FIX-3] ARIMA không nhận enforce_* -> không truyền
                model = ARIMA(
                    y_train,
                    exog=exog_train,
                    order=self.order,
                )

            self.model_fitted = model.fit(disp=False, maxiter=300)

        except Exception as e:
            warnings.warn(
                f"[{self.name}] fit() gap loi ({e}). "
                "Chuyen sang fallback ARIMA(1,1,1)..."
            )
            # Fallback don gian hon, van giu y_train de naive forecast co the dung
            try:
                fb = ARIMA(y_train, order=(1, 1, 1))
                self.model_fitted = fb.fit(disp=False)
            except Exception:
                self.model_fitted = None  # hoan toan that bai -> naive forecast

        return self

    # ------------------------------------------------------------------
    def predict(
        self,
        horizon: int = 28,
        exog_future: Optional[np.ndarray] = None,
        return_conf_int: bool = True,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Dự báo H bước thời gian tiếp theo.

        Returns:
            (forecast_mean, ci_lower, ci_upper)
            ci_lower / ci_upper là None khi không tính được CI.
        """
        if self.model_fitted is not None:
            try:
                fcast = self.model_fitted.get_forecast(
                    steps=horizon, exog=exog_future
                )
                preds = np.clip(np.asarray(fcast.predicted_mean), 0, None)

                lower_b = upper_b = None
                if return_conf_int:
                    # [FIX-2] conf_int() tra ve DataFrame; dung .values
                    ci = fcast.conf_int(alpha=alpha).values
                    lower_b = np.clip(ci[:, 0], 0, None)
                    upper_b = np.clip(ci[:, 1], 0, None)

                return preds, lower_b, upper_b

            except Exception as e:
                warnings.warn(
                    f"[{self.name}] predict() gap loi ({e}). "
                    "Chuyen sang naive seasonal fallback..."
                )

        # [FIX-1] Naive seasonal fallback: dung rolling mean 7 ngay cuoi cua
        # y_train, dem lai theo chu ky de tranh bias "gia tri cuoi cung".
        preds = self._naive_seasonal_forecast(horizon)
        return preds, None, None

    # ------------------------------------------------------------------
    def _naive_seasonal_forecast(self, horizon: int, period: int = 7) -> np.ndarray:
        """
        Fallback: lấy trung bình của 4 tuần cuối cùng cùng ngày trong tuần
        (tương đương Seasonal Naive với làm trơn nhẹ).
        """
        if self._y_train is None or len(self._y_train) == 0:
            return np.zeros(horizon)

        preds = np.zeros(horizon)
        n = len(self._y_train)
        num_avg = 4  # số tuần để tính trung bình

        for h in range(horizon):
            # Vị trí ngày tương ứng trong chuỗi lịch sử (cùng vị trí trong tuần)
            indices = [n - period * k + h for k in range(num_avg, 0, -1)]
            valid_vals = [
                self._y_train[i] for i in indices
                if 0 <= i < n
            ]
            preds[h] = float(np.mean(valid_vals)) if valid_vals else 0.0

        return np.clip(preds, 0, None)
