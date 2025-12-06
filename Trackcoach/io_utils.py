"""CSV reading, column name normalization, and unit standardization module."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Column name alias sets
TIME_ALIASES = [
    "t_s", "time_s", "time", "timestamp", "Timestamp",
    "elapsed_time", "elapsed_time (s)"
]

LAP_ALIASES = [
    "lap", "Lap", "Lap #", "lap_number"
]

LAT_ALIASES = [
    "lat", "latitude", "Latitude", "latitude (deg)", "Latitude (deg)",
    "latitude_deg", "Latitude_deg"
]

LON_ALIASES = [
    "lon", "longitude", "Longitude", "longitude (deg)", "Longitude (deg)",
    "longitude_deg", "Longitude_deg"
]

SPEED_ALIASES = [
    "speed", "speed (m/s)", "Speed (m/s)", "speed_mps", "speed_m/s",
    "speed_kmh", "speed (km/h)", "Speed (km/h)",
    "speed_mph", "speed (mph)", "Speed (mph)"
]

LONGITUDINAL_ACC_ALIASES = [
    "longitudinal_acc", "longitudinal_acc (g)", "longitudinal g",
    "longitudinal-acc (g)"
]


def normalize_column_name(col: str) -> str:
    """
    Normalize column names: lowercase, strip whitespace, compress consecutive whitespace to single space.

    Args:
        col: Original column name

    Returns:
        Normalized column name
    """
    col = col.lower()
    col = col.strip()
    col = re.sub(r'\s+', ' ', col)
    return col


def normalize_for_matching(col: str) -> str:
    """
    Normalization for matching: remove parentheses and their contents, further compress whitespace.

    Args:
        col: Column name

    Returns:
        Normalized column name for matching
    """
    col = normalize_column_name(col)
    col = re.sub(r'\([^)]*\)', '', col)  # Remove parentheses and contents
    col = re.sub(r'\s+', '', col)  # Remove all whitespace
    return col


def find_column_by_aliases(
    df: pd.DataFrame,
    aliases: List[str],
    override: Optional[str] = None
) -> Optional[str]:
    """
    Find column name by alias set.

    Args:
        df: DataFrame
        aliases: Alias list
        override: CLI override value (takes priority)

    Returns:
        Found column name, or None
    """
    if override:
        override_norm = normalize_column_name(override)
        for col in df.columns:
            if normalize_column_name(col) == override_norm:
                logger.info(f"Using CLI override column: {col}")
                return col
        logger.warning(f"CLI override column '{override}' not found, falling back to auto-detection")

    # Build normalization mapping
    col_map = {normalize_for_matching(col): col for col in df.columns}
    alias_map = {normalize_for_matching(alias): alias for alias in aliases}

    for alias_norm, alias_orig in alias_map.items():
        if alias_norm in col_map:
            found_col = col_map[alias_norm]
            logger.info(f"Matched column '{found_col}' to alias '{alias_orig}'")
            return found_col

    return None


def read_csv_with_encoding(file_path: Path) -> pd.DataFrame:
    """
    Try multiple encodings to read CSV file.

    Args:
        file_path: CSV file path

    Returns:
        Read DataFrame

    Raises:
        ValueError: Raised when all encodings fail
    """
    encodings = ['utf-8', 'utf-8-sig', 'latin-1']
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            logger.info(f"Successfully read CSV with encoding: {enc}")
            return df
        except UnicodeDecodeError as e:
            last_error = e
            logger.debug(f"Failed to read with encoding {enc}: {e}")
            continue

    raise ValueError(
        f"Failed to read CSV file '{file_path}' with encodings {encodings}. "
        f"Last error: {last_error}. Please check file encoding or convert to UTF-8."
    )


def detect_speed_unit(df: pd.DataFrame, speed_col: str) -> str:
    """
    Detect speed column unit.

    Args:
        df: DataFrame
        speed_col: Speed column name

    Returns:
        Unit identifier: 'ms', 'kmh', 'mph', or 'unknown'
    """
    col_norm = normalize_for_matching(speed_col)
    
    if 'mph' in col_norm or 'mph' in speed_col.lower():
        return 'mph'
    elif 'kmh' in col_norm or 'km/h' in speed_col.lower():
        return 'kmh'
    elif 'mps' in col_norm or 'm/s' in speed_col.lower():
        return 'ms'
    else:
        # Try to infer from value range (not precise, for reference only)
        if speed_col in df.columns:
            # Convert to numeric first (handle string types)
            speed_series = pd.to_numeric(df[speed_col], errors='coerce')
            if len(speed_series.dropna()) > 0:
                max_val = speed_series.max()
                if pd.notna(max_val):
                    if max_val > 100:  # mayiskm/hormph
                        return 'unknown'
                    elif max_val < 50:  # mayism/s
                        return 'ms'
        return 'unknown'


def convert_speed_to_ms(df: pd.DataFrame, speed_col: str, unit: str) -> pd.Series:
    """
    Convert speed to m/s.

    Args:
        df: DataFrame
        speed_col: Speed column name
        unit: Unit ('ms', 'kmh', 'mph')

    Returns:
        Converted velocity series (m/s)
    """
    speed = df[speed_col].copy()
    
    if unit == 'kmh':
        speed = speed / 3.6
        logger.info(f"Converted speed from km/h to m/s (divided by 3.6)")
    elif unit == 'mph':
        speed = speed * 0.44704
        logger.info(f"Converted speed from mph to m/s (multiplied by 0.44704)")
    elif unit == 'ms':
        logger.info(f"Speed already in m/s")
    else:
        logger.warning(f"Unknown speed unit '{unit}', assuming m/s")
    
    return speed


def process_lap_column(df: pd.DataFrame, lap_col: Optional[str]) -> pd.Series:
    """
    Process lap number column: forward fill, fill 0, convert to integer.

    Args:
        df: DataFrame
        lap_col: Lap number column name

    Returns:
        Processed lap number series
    """
    if lap_col is None or lap_col not in df.columns:
        logger.info("No lap column found, creating default lap=0")
        return pd.Series(0, index=df.index)
    
    lap = df[lap_col].copy()
    
    # Count missing values
    missing_before = lap.isna().sum()
    
    # Forward fill
    lap = lap.ffill()
    missing_after_ffill = lap.isna().sum()
    
    # Fill 0
    lap = lap.fillna(0)
    
    # Convert to integer
    lap = lap.astype(int)
    
    missing_after_fillna = (lap == 0).sum() if missing_before > 0 else 0
    
    logger.info(
        f"Lap column processing: missing_before={missing_before}, "
        f"missing_after_ffill={missing_after_ffill}, "
        f"zeros_after_fillna={missing_after_fillna}"
    )
    
    return lap


def read_and_preprocess_csv(
    csv_path: Path,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    time_col: Optional[str] = None,
    speed_col: Optional[str] = None,
    lap_col: Optional[str] = None,
    along_col: Optional[str] = None,
    speed_unit: str = "auto"
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Read and preprocess CSV file.

    Args:
        csv_path: CSV file path
        lat_col: CLI override latitude column
        lon_col: CLI override longitude column
        time_col: CLI override time column
        speed_col: CLI override speed column
        lap_col: CLI override lap number column
        along_col: CLI override longitudinal acceleration column
        speed_unit: Speed unit ('auto', 'ms', 'kmh', 'mph')

    Returns:
        (Preprocessed DataFrame, column name mapping dictionary)

    Raises:
        ValueError: Raised when required columns are missing
    """
    # Read CSV
    df = read_csv_with_encoding(csv_path)
    
    logger.info(f"Read CSV with {len(df)} rows and {len(df.columns)} columns")
    
    # Normalize all column names (for matching)
    col_map = {}  # Original column name -> normalized column names
    
    # Find required columns
    time_col_found = find_column_by_aliases(df, TIME_ALIASES, time_col)
    lat_col_found = find_column_by_aliases(df, LAT_ALIASES, lat_col)
    lon_col_found = find_column_by_aliases(df, LON_ALIASES, lon_col)
    speed_col_found = find_column_by_aliases(df, SPEED_ALIASES, speed_col)
    
    if time_col_found is None:
        raise ValueError(
            f"Time column not found. Expected aliases: {TIME_ALIASES}. "
            f"Available columns: {list(df.columns)}. "
            f"Use --time-col to specify manually."
        )
    
    if lat_col_found is None:
        raise ValueError(
            f"Latitude column not found. Expected aliases: {LAT_ALIASES}. "
            f"Available columns: {list(df.columns)}. "
            f"Use --lat-col to specify manually."
        )
    
    if lon_col_found is None:
        raise ValueError(
            f"Longitude column not found. Expected aliases: {LON_ALIASES}. "
            f"Available columns: {list(df.columns)}. "
            f"Use --lon-col to specify manually."
        )
    
    if speed_col_found is None:
        raise ValueError(
            f"Speed column not found. Expected aliases: {SPEED_ALIASES}. "
            f"Available columns: {list(df.columns)}. "
            f"Use --speed-col to specify manually."
        )
    
    # If velocity column found, check if multiple velocity columns exist, prioritize m/s column
    if speed_unit == "auto" and speed_col is None:  # Only optimize in auto-detection mode
        speed_cols = [c for c in df.columns if any(alias in normalize_for_matching(c) 
                                                   for alias in ['speed', 'velocity'])]
        if len(speed_cols) > 1:
            # Prioritize m/s column
            for col in speed_cols:
                col_norm = normalize_for_matching(col)
                if 'm/s' in col.lower() or ('mps' in col_norm and 'mph' not in col.lower()):
                    speed_col_found = col
                    logger.info(f"Multiple speed columns found, prioritizing m/s column: {col}")
                    break
    
    # Find optional columns
    lap_col_found = find_column_by_aliases(df, LAP_ALIASES, lap_col)
    along_col_found = find_column_by_aliases(df, LONGITUDINAL_ACC_ALIASES, along_col)
    
    # Detect velocity unit
    if speed_unit == "auto":
        detected_unit = detect_speed_unit(df, speed_col_found)
        if detected_unit == "unknown":
            detected_unit = 'ms'  # Default assumption
        speed_unit = detected_unit
    
    # Convert velocity
    v = convert_speed_to_ms(df, speed_col_found, speed_unit)
    
    # Process time column
    t_s = df[time_col_found].copy()
    # If time column is string timestamp, need conversion (simplified here, assume already in seconds)
    if t_s.dtype == 'object':
        logger.warning(f"Time column '{time_col_found}' is object type, attempting conversion")
        try:
            t_s = pd.to_numeric(t_s, errors='coerce')
        except Exception as e:
            logger.warning(f"Could not convert time column to numeric: {e}")
    
    # Process lap number column
    lap = process_lap_column(df, lap_col_found)
    
    # Convert lat/lon to numeric (handle string types)
    lat_series = pd.to_numeric(df[lat_col_found], errors='coerce')
    lon_series = pd.to_numeric(df[lon_col_found], errors='coerce')
    
    # Log conversion warnings if needed
    if lat_series.isna().any() and not df[lat_col_found].isna().any():
        logger.warning(f"Some latitude values could not be converted to numeric, converted to NaN")
    if lon_series.isna().any() and not df[lon_col_found].isna().any():
        logger.warning(f"Some longitude values could not be converted to numeric, converted to NaN")
    
    # Build output DataFrame
    result_df = pd.DataFrame({
        't_s': t_s,
        'lat': lat_series,
        'lon': lon_series,
        'v': v,
        'lap': lap
    })
    
    # Add longitudinal acceleration (if exists)
    if along_col_found:
        result_df['longitudinal_acc_g'] = df[along_col_found]
        logger.info(f"Found longitudinal acceleration column: {along_col_found}")
    else:
        result_df['longitudinal_acc_g'] = None
        logger.info("No longitudinal acceleration column found, will use dv/dt")
    
    # Add other optional columns (preserve original data)
    optional_cols = ['rpm', 'throttle', 'throttle_pos']
    for col in optional_cols:
        if col in df.columns:
            result_df[col] = df[col]
    
    # Check altitude column (if all zeros then ignore)
    if 'altitude' in df.columns or 'altitude (m)' in df.columns:
        alt_col = 'altitude (m)' if 'altitude (m)' in df.columns else 'altitude'
        if (df[alt_col] == 0).all():
            logger.info(f"Altitude column '{alt_col}' is all zeros, ignoring")
        else:
            result_df['altitude'] = df[alt_col]
    
    # Remove missing value rows (only for required columns)
    missing_mask = result_df[['t_s', 'lat', 'lon', 'v']].isna().any(axis=1)
    if missing_mask.any():
        logger.warning(f"Removing {missing_mask.sum()} rows with missing essential columns")
        result_df = result_df[~missing_mask].reset_index(drop=True)
    
    # Column name mapping
    col_mapping = {
        'time': time_col_found,
        'lat': lat_col_found,
        'lon': lon_col_found,
        'speed': speed_col_found,
        'lap': lap_col_found if lap_col_found else None,
        'along': along_col_found if along_col_found else None
    }
    
    logger.info(f"Preprocessed data: {len(result_df)} rows")
    logger.info(f"Speed unit: {speed_unit}, converted to m/s")
    
    return result_df, col_mapping

