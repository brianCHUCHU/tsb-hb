"""
Hurdle Model for Intermittent Demand Forecasting

Two-stage model:
1. Occurrence: Logistic regression for P(demand > 0)
2. Size: Log-normal model for demand size given demand > 0

This serves as a baseline comparison to TSB-HB.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class HurdleModel:
    """
    Hurdle Model for intermittent demand forecasting.
    
    Stage 1: Models probability of non-zero demand using logistic regression
    Stage 2: Models demand size using log-normal distribution (MLE)
    
    This is a per-series model (no pooling across items).
    """
    
    def __init__(self):
        self.p_models = {}  # Per-series occurrence probability models
        self.size_params = {}  # Per-series size distribution parameters (mu, sigma)
        
    def fit(self, train_df: pd.DataFrame) -> HurdleModel:
        """
        Fit the Hurdle Model on training data.
        
        Args:
            train_df: DataFrame with columns ['unique_id', 'ds', 'y']
        
        Returns:
            self
        """
        for uid in train_df['unique_id'].unique():
            series = train_df[train_df['unique_id'] == uid].copy()
            series = series.sort_values('ds')
            
            # Extract demand values
            y = series['y'].values
            
            # Stage 1: Occurrence model
            # For simplicity, use empirical probability (could use time features for logistic regression)
            n_total = len(y)
            n_positive = np.sum(y > 0)
            p_hat = n_positive / n_total if n_total > 0 else 0.0
            
            self.p_models[uid] = p_hat
            
            # Stage 2: Size model (log-normal MLE)
            positive_demands = y[y > 0]
            
            if len(positive_demands) > 0:
                log_demands = np.log(positive_demands)
                mu_hat = np.mean(log_demands)
                sigma_hat = np.std(log_demands, ddof=1) if len(log_demands) > 1 else 1e-6
                
                # Store parameters
                self.size_params[uid] = {
                    'mu': mu_hat,
                    'sigma': max(sigma_hat, 1e-6)  # Avoid zero variance
                }
            else:
                # No positive demands observed
                self.size_params[uid] = {
                    'mu': 0.0,
                    'sigma': 1.0
                }
        
        return self
    
    def predict(
        self, 
        eval_df: pd.DataFrame, 
        quantiles: Optional[list[float]] = None,
        n_samples: int = 2000
    ) -> pd.DataFrame:
        """
        Generate predictions for evaluation data.
        
        Args:
            eval_df: DataFrame with columns ['unique_id', 'ds']
            quantiles: If provided, return probabilistic forecasts. 
                      If None, return point forecasts.
            n_samples: Number of Monte Carlo samples for probabilistic forecasts
        
        Returns:
            DataFrame with predictions
        """
        if quantiles is None or len(quantiles) == 0:
            # Point forecast: E[Y] = P(Y > 0) * E[Y | Y > 0]
            return self._predict_point(eval_df)
        else:
            # Probabilistic forecast via Monte Carlo
            return self._predict_probabilistic(eval_df, quantiles, n_samples)
    
    def _predict_point(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """Generate point forecasts."""
        predictions = []
        
        for uid in eval_df['unique_id'].unique():
            if uid not in self.p_models:
                continue
                
            p = self.p_models[uid]
            mu = self.size_params[uid]['mu']
            sigma_sq = self.size_params[uid]['sigma'] ** 2
            
            # E[Y] = P(Y > 0) * E[Y | Y > 0]
            # For log-normal: E[Y | Y > 0] = exp(mu + sigma^2/2)
            expected_size = np.exp(mu + sigma_sq / 2)
            yhat = p * expected_size
            
            series_dates = eval_df[eval_df['unique_id'] == uid]['ds'].values
            predictions.append(pd.DataFrame({
                'unique_id': uid,
                'ds': series_dates,
                'yhat': yhat
            }))
        
        if not predictions:
            return pd.DataFrame(columns=['unique_id', 'ds', 'yhat'])
        
        return pd.concat(predictions, ignore_index=True)
    
    def _predict_probabilistic(
        self, 
        eval_df: pd.DataFrame, 
        quantiles: list[float],
        n_samples: int
    ) -> pd.DataFrame:
        """Generate probabilistic forecasts via Monte Carlo sampling."""
        predictions = []
        
        for uid in eval_df['unique_id'].unique():
            if uid not in self.p_models:
                continue
            
            p = self.p_models[uid]
            mu = self.size_params[uid]['mu']
            sigma = self.size_params[uid]['sigma']
            
            # Monte Carlo sampling
            # 1. Sample occurrence: Bernoulli(p)
            occurs = np.random.binomial(1, p, n_samples)
            
            # 2. Sample size: Log-Normal(mu, sigma)
            sizes = np.exp(np.random.normal(mu, sigma, n_samples))
            
            # 3. Combine: Y = occurs * size
            samples = occurs * sizes
            
            # Compute quantiles
            qvals = {f'q_{q}': np.quantile(samples, q) for q in quantiles}
            qvals['prob_zero_predicted'] = 1.0 - p
            
            series_dates = eval_df[eval_df['unique_id'] == uid]['ds'].values
            predictions.append(pd.DataFrame({
                **qvals,
                'unique_id': uid,
                'ds': series_dates
            }))
        
        if not predictions:
            cols = ['unique_id', 'ds'] + [f'q_{q}' for q in quantiles] + ['prob_zero_predicted']
            return pd.DataFrame(columns=cols)
        
        return pd.concat(predictions, ignore_index=True)


def fit_predict_hurdle(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    quantiles: Optional[list[float]] = None,
    n_samples: int = 2000
) -> pd.DataFrame:
    """
    Convenience function to fit and predict with Hurdle Model.
    
    Args:
        train_df: Training data with ['unique_id', 'ds', 'y']
        eval_df: Evaluation data with ['unique_id', 'ds']
        quantiles: If provided, return probabilistic forecasts
        n_samples: Number of samples for probabilistic forecasts
    
    Returns:
        Predictions DataFrame
    """
    model = HurdleModel()
    model.fit(train_df)
    predictions = model.predict(eval_df, quantiles=quantiles, n_samples=n_samples)
    
    return predictions
