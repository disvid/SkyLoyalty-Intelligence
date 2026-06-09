"""
src/utils.py
------------
Shared utility functions: logging setup, directory creation,
plotting helpers, and timing decorators.
"""

import os
import time
import logging
import functools
from typing import Any, Callable

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers

from config import LOG_LEVEL, LOG_FORMAT, PLOTS_DIR, MODELS_DIR, REPORTS_DIR


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger with console handler.

    Parameters
    ----------
    name : str
        Logger name (usually __name__).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL))
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    return logger


def ensure_dirs() -> None:
    """
    Create all output directories if they don't exist.
    Call once at the start of any pipeline run.
    """
    for directory in [PLOTS_DIR, MODELS_DIR, REPORTS_DIR]:
        os.makedirs(directory, exist_ok=True)


def timer(func: Callable) -> Callable:
    """
    Decorator that logs execution time of a function.

    Parameters
    ----------
    func : Callable
        Function to wrap.
    """
    logger = get_logger("timer")

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result

    return wrapper


def save_figure(fig: plt.Figure, filename: str, dpi: int = 150) -> str:
    """
    Save a matplotlib figure to the plots directory.

    Parameters
    ----------
    fig : plt.Figure
    filename : str  — without extension
    dpi : int

    Returns
    -------
    str  — full path where figure was saved
    """
    path = os.path.join(PLOTS_DIR, f"{filename}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def value_counts_pct(series: Any) -> Any:
    """
    Returns value counts with percentage column.

    Parameters
    ----------
    series : pd.Series

    Returns
    -------
    pd.DataFrame
    """
    import pandas as pd
    vc = series.value_counts()
    pct = (vc / vc.sum() * 100).round(2)
    return pd.DataFrame({"count": vc, "pct": pct})