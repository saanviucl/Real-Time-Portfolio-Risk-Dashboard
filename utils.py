
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def percent(x: float) -> str:
    return f"{x*100:.2f}%"

def annualize_return(daily_mean: float, periods_per_year: int = 252) -> float:
    return (1 + daily_mean)**periods_per_year - 1

def annualize_vol(daily_std: float, periods_per_year: int = 252) -> float:
    return daily_std * (periods_per_year**0.5)

def make_heatmap(ax, data: pd.DataFrame, title: str, xlabel: str, ylabel: str):
    # Use matplotlib 
    im = ax.imshow(data.values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index)
    # annotate
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data.values[i, j]:.2f}", ha="center", va="center")
    return im

def contrib_to_risk(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    # Marginal risk contribution = (Sigma w) / portfolio_vol
    port_var = float(weights.T @ cov.values @ weights)
    if port_var <= 0:
        return pd.Series(np.zeros_like(weights), index=weights.index)
    mrc = cov.values @ weights  # marginal variance contributions
    cr = weights * mrc / (port_var**0.5)  # contribution to vol
    return pd.Series(cr, index=weights.index)
