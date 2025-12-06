"""Velocity smoothing, acceleration, and curvature computation module."""

import logging
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from trackcoach.geom import resample_by_arc_length

logger = logging.getLogger(__name__)

G = 9.81  # Gravitational acceleration constant


def estimate_sampling_interval(t_s: pd.Series) -> float:
    """
    Estimate sampling interval (median).

    Args:
        t_s: Time series (seconds)

    Returns:
        Sampling interval (seconds)
    """
    dt = t_s.diff().median()
    if pd.isna(dt) or dt <= 0:
        dt = t_s.diff().mean()
    if pd.isna(dt) or dt <= 0:
        dt = 0.1  # Default value
        logger.warning(f"Could not estimate sampling interval, using default {dt}s")
    else:
        logger.info(f"Estimated sampling interval: {dt:.4f} s")
    return dt


def smooth_velocity(
    v: pd.Series,
    t_s: pd.Series,
    smooth_win_sec: float = 0.5
) -> pd.Series:
    """
    Smooth velocity using Savitzky-Golay filter.

    Args:
        v: Velocity series (m/s)
        t_s: Time series (seconds)
        smooth_win_sec: Smoothing window length (seconds)

    Returns:
        Smoothed velocity series
    """
    dt = estimate_sampling_interval(t_s)
    
    # Compute window points (odd number, at least 5)
    window_length = int(smooth_win_sec / dt)
    if window_length < 5:
        window_length = 5
    elif window_length % 2 == 0:
        window_length += 1
    
    if len(v) < window_length:
        logger.warning(f"Data length ({len(v)}) < window length ({window_length}), skipping smoothing")
        return v
    
    v_smooth = savgol_filter(v.values, window_length, polyorder=2)
    v_smooth = pd.Series(v_smooth, index=v.index)
    
    logger.info(f"Smoothed velocity with window length={window_length} points ({smooth_win_sec:.2f}s)")
    
    return v_smooth


def compute_longitudinal_acceleration(
    v_smooth: pd.Series,
    t_s: pd.Series,
    longitudinal_acc_g: Optional[pd.Series] = None
) -> Tuple[pd.Series, Literal["sensor_g", "dvdt"]]:
    """
    Compute longitudinal acceleration.

    Priority: sensor acceleration (G) > numerical derivative.

    Args:
        v_smooth: Smoothed velocity series (m/s)
        t_s: Time series (seconds)
        longitudinal_acc_g: Sensor longitudinal acceleration (G, optional)

    Returns:
        (Longitudinal acceleration series (m/s²), data source identifier)
    """
    if longitudinal_acc_g is not None and not longitudinal_acc_g.isna().all():
        # Use sensor data
        a_long = longitudinal_acc_g * G
        
        # Light smoothing
        dt = estimate_sampling_interval(t_s)
        window_length = max(5, int(0.2 / dt))  # Approximately 0.2 seconds window
        if window_length % 2 == 0:
            window_length += 1
        if len(a_long) >= window_length:
            a_long = pd.Series(
                savgol_filter(a_long.values, window_length, polyorder=2),
                index=a_long.index
            )
        
        source = "sensor_g"
        logger.info(f"Using sensor longitudinal acceleration (G), converted to m/s²")
    else:
        # Use numerical derivative
        dt = estimate_sampling_interval(t_s)
        a_long = v_smooth.diff() / dt
        source = "dvdt"
        logger.info(f"Using numerical derivative for longitudinal acceleration (dv/dt)")
    
    logger.info(f"Longitudinal acceleration source: {source}")
    
    return a_long, source


def compute_jerk(
    a_long: pd.Series,
    t_s: pd.Series
) -> pd.Series:
    """
    Compute jerk (jerk = da/dt).

    Args:
        a_long: Longitudinal acceleration series (m/s²)
        t_s: Time series (seconds)

    Returns:
        Jerk series (m/s³)
    """
    dt = estimate_sampling_interval(t_s)
    jerk = a_long.diff() / dt
    
    logger.info(f"Computed jerk (da/dt)")
    
    return jerk


def compute_curvature(
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray
) -> np.ndarray:
    """
    Compute curvature κ(s) = |x'y'' - y'x''| / (x'² + y'²)^(3/2).

    Args:
        x: X coordinate array (parameterized by arc length)
        y: Y coordinate array (parameterized by arc length)
        s: Arc length array

    Returns:
        Curvature array κ(s)
    """
    # Compute first derivative
    dx_ds = np.gradient(x, s)
    dy_ds = np.gradient(y, s)
    
    # Compute second derivative
    d2x_ds2 = np.gradient(dx_ds, s)
    d2y_ds2 = np.gradient(dy_ds, s)
    
    # Compute curvature
    numerator = np.abs(dx_ds * d2y_ds2 - dy_ds * d2x_ds2)
    denominator = (dx_ds**2 + dy_ds**2)**(3/2)
    
    # Avoid division by zero
    denominator = np.where(denominator < 1e-10, 1e-10, denominator)
    kappa = numerator / denominator
    
    logger.info(f"Computed curvature: min={kappa.min():.6f}, max={kappa.max():.6f}, mean={kappa.mean():.6f}")
    
    return kappa


def smooth_curvature(
    kappa: np.ndarray,
    window_length: int = 21
) -> np.ndarray:
    """
    Smooth curvature (Savitzky-Golay).

    Args:
        kappa: Curvature array
        window_length: Window length (odd number)

    Returns:
        Smoothed curvature array
    """
    if len(kappa) < window_length:
        logger.warning(f"Curvature length ({len(kappa)}) < window length ({window_length}), skipping smoothing")
        return kappa
    
    if window_length % 2 == 0:
        window_length += 1
    
    kappa_smooth = savgol_filter(kappa, window_length, polyorder=3)
    
    logger.info(f"Smoothed curvature with window length={window_length}")
    
    return kappa_smooth


def build_reference_centerline(
    df: pd.DataFrame,
    ref_lap_idx: Optional[int],
    n_points: int = 1000,
    x_col: str = 'x',
    y_col: str = 'y',
    s_col: str = 's',
    lap_col: str = 'lap'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build reference centerline using specified lap data with uniform resampling.
    
    Args:
        df: DataFrame, must contain x, y, s columns
        ref_lap_idx: Reference lap number (None uses all data)
        n_points: Resampling point count
        x_col: X coordinate column name
        y_col: Y coordinate column name
        s_col: Arc length column name
        lap_col: Lap number column name
    
    Returns:
        (s_ref, x_ref, y_ref) Uniformly resampled reference centerline
    """
    if x_col not in df.columns or y_col not in df.columns or s_col not in df.columns:
        raise ValueError("DataFrame must contain x, y, and s columns to build reference centerline")
    
    if ref_lap_idx is not None and lap_col in df.columns:
        lap_df = df[df[lap_col] == ref_lap_idx].copy()
        if lap_df.empty:
            logger.warning(f"Reference lap #{ref_lap_idx} not found; using entire dataset for centerline")
            lap_df = df.copy()
    else:
        lap_df = df.copy()
    
    lap_df = lap_df.sort_values(by=s_col)
    
    if len(lap_df) < 3:
        raise ValueError("Insufficient points to build reference centerline")
    
    s_ref, x_ref, y_ref = resample_by_arc_length(
        lap_df[x_col],
        lap_df[y_col],
        lap_df[s_col],
        n_points=n_points
    )
    
    logger.info(
        f"Built reference centerline from lap {ref_lap_idx if ref_lap_idx is not None else 'ALL'} "
        f"with {len(lap_df)} samples -> {n_points} resampled points"
    )
    
    return s_ref, x_ref, y_ref



