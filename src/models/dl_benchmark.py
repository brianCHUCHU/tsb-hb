from __future__ import annotations

import pandas as pd


class DLBenchmark:
    """Placeholder DL benchmark interface.

    Implementors should provide model initialization, training, and prediction
    that match the following API. For probabilistic forecasts, return a DataFrame
    with quantile columns named like q_0.1, q_0.5, q_0.9, etc.
    """

    def fit(self, train_df: pd.DataFrame) -> "DLBenchmark":
        # Implement: use train_df with columns [unique_id, ds, y]
        return self

    def predict(self, test_df: pd.DataFrame, quantiles: list[float] | None = None) -> pd.DataFrame:
        # Implement: return DataFrame with [unique_id, ds, yhat] or quantiles
        raise NotImplementedError

