"""Coordinate projection, arc length computation, and resampling module."""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def build_local_projection(lon0: float, lat0: float) -> str:
    """
    Build local transverse Mercator projection string.

    Args:
        lon0: Center longitude
        lat0: Center latitude

    Returns:
        PROJ string
    """
    proj_str = (
        f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} "
        f"+k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return proj_str


def project_to_local_coords(
    lon: pd.Series,
    lat: pd.Series,
    lon0: Optional[float] = None,
    lat0: Optional[float] = None
) -> Tuple[pd.Series, pd.Series, float, float]:
    """
    Project WGS84 longitude/latitude to local metric coordinates.

    Args:
        lon: Longitude series
        lat: Latitude series
        lon0: Center longitude (if None, use median)
        lat0: Center latitude (if None, use median)

    Returns:
        (X coordinate series, Y coordinate series, actually used lon0, actually used lat0)
    """
    if lon0 is None:
        lon0 = float(lon.median())
    if lat0 is None:
        lat0 = float(lat.median())
    
    logger.info(f"Projection center: lon0={lon0:.6f}, lat0={lat0:.6f}")
    
    # Build projection
    proj_str = build_local_projection(lon0, lat0)
    transformer = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
    
    # Project
    x, y = transformer.transform(lon.values, lat.values)
    
    logger.info(f"Projected coordinates: x range [{x.min():.2f}, {x.max():.2f}], "
                f"y range [{y.min():.2f}, {y.max():.2f}]")
    
    return pd.Series(x, index=lon.index), pd.Series(y, index=lat.index), lon0, lat0


def compute_arc_length(
    x: pd.Series,
    y: pd.Series,
    lap: Optional[pd.Series] = None
) -> pd.Series:
    """
    Compute arc length (cumulative Euclidean distance between adjacent points).

    Args:
        x: X coordinate series
        y: Y coordinate series
        lap: Lap number series (optional, if exists then compute separately per lap starting from 0)

    Returns:
        Arc length series (meters)
    """
    if lap is None:
        # Global computation
        dx = x.diff()
        dy = y.diff()
        ds = np.sqrt(dx**2 + dy**2)
        s = ds.cumsum().fillna(0)
    else:
        # Compute per lap
        s = pd.Series(0.0, index=x.index)
        for lap_num in lap.unique():
            mask = lap == lap_num
            x_lap = x[mask]
            y_lap = y[mask]
            
            if len(x_lap) < 2:
                continue
            
            dx = x_lap.diff()
            dy = y_lap.diff()
            ds = np.sqrt(dx**2 + dy**2)
            s_lap = ds.cumsum().fillna(0)
            s[mask] = s_lap.values
    
    logger.info(f"Arc length computed: max={s.max():.2f} m")
    
    return s


def resample_by_arc_length(
    x: pd.Series,
    y: pd.Series,
    s: pd.Series,
    n_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample uniformly by arc length.

    Args:
        x: X coordinate series
        y: Y coordinate series
        s: Arc length series
        n_points: Target number of points

    Returns:
        (s_new, x_new, y_new) - New arc length, X, Y arrays
    """
    # RemoveNaN
    valid_mask = ~(s.isna() | x.isna() | y.isna())
    s_valid = s[valid_mask].values
    x_valid = x[valid_mask].values
    y_valid = y[valid_mask].values
    
    if len(s_valid) < 2:
        raise ValueError("Not enough valid points for resampling")
    
    # Create new uniform arc length grid
    s_new = np.linspace(s_valid.min(), s_valid.max(), n_points)
    
    # Interpolate
    f_x = interp1d(s_valid, x_valid, kind='linear', bounds_error=False, fill_value='extrapolate')
    f_y = interp1d(s_valid, y_valid, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    x_new = f_x(s_new)
    y_new = f_y(s_new)
    
    logger.info(f"Resampled to {n_points} points, arc length range: [{s_new.min():.2f}, {s_new.max():.2f}]")
    
    return s_new, x_new, y_new










