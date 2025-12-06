"""Configuration management module: parameter presets and config file management."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default parameter presets
DEFAULT_PRESETS = {
    'default': {
        'description': 'Default parameter configuration',
        'smooth_win_sec': 0.5,
        'apex_height_quantile': 0.78,
        'min_turn_gap_m': 65.0,
        'apex_prom_mult': 0.60,
        'post_merge_gap_m': 45.0,
        'segment_window_m': 60.0,
        'brake_thr_g': -0.28,
        'brake_onset_thr_g': -0.15,
        'late_brake_thr_m': 7.0,
        'early_brake_thr_m': 12.0,
        'overslow_thr_ms': 1.2,
        'weak_exit_thr_ms': 1.5,
        'weak_exit_adecel_g': 0.25,
        'stability_weight': 0.3,
        'topk': 8
    },
    'sows_track': {
        'description': 'SOWS track preset (12-16 turns)',
        'smooth_win_sec': 0.5,
        'apex_height_quantile': 0.75,
        'min_turn_gap_m': 65.0,
        'apex_prom_mult': 0.60,
        'post_merge_gap_m': 45.0,
        'segment_window_m': 60.0,
        'brake_thr_g': -0.28,
        'brake_onset_thr_g': -0.15,
        'late_brake_thr_m': 7.0,
        'early_brake_thr_m': 12.0,
        'overslow_thr_ms': 1.2,
        'weak_exit_thr_ms': 1.5,
        'weak_exit_adecel_g': 0.25,
        'stability_weight': 0.3,
        'topk': 8
    },
    'high_speed_track': {
        'description': 'High-speed track preset',
        'smooth_win_sec': 0.5,
        'apex_height_quantile': 0.80,
        'min_turn_gap_m': 75.0,
        'apex_prom_mult': 0.70,
        'post_merge_gap_m': 50.0,
        'segment_window_m': 70.0,
        'brake_thr_g': -0.30,
        'brake_onset_thr_g': -0.18,
        'late_brake_thr_m': 8.0,
        'early_brake_thr_m': 15.0,
        'overslow_thr_ms': 1.5,
        'weak_exit_thr_ms': 2.0,
        'weak_exit_adecel_g': 0.30,
        'stability_weight': 0.25,
        'topk': 10
    },
    'technical_track': {
        'description': 'Technical track preset (multiple turns)',
        'smooth_win_sec': 0.5,
        'apex_height_quantile': 0.70,
        'min_turn_gap_m': 55.0,
        'apex_prom_mult': 0.55,
        'post_merge_gap_m': 40.0,
        'segment_window_m': 55.0,
        'brake_thr_g': -0.25,
        'brake_onset_thr_g': -0.12,
        'late_brake_thr_m': 6.0,
        'early_brake_thr_m': 10.0,
        'overslow_thr_ms': 1.0,
        'weak_exit_thr_ms': 1.2,
        'weak_exit_adecel_g': 0.20,
        'stability_weight': 0.35,
        'topk': 12
    }
}


class ConfigManager:
    """Configuration manager: load and manage parameter presets."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Config file path, if None then use default presets
        """
        self.config_file = config_file
        self.presets = DEFAULT_PRESETS.copy()
        self.custom_config: Dict[str, Any] = {}
        
        if config_file and config_file.exists():
            self.load_config(config_file)
    
    def load_config(self, config_file: Path) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.custom_config = json.load(f)
            
            # If config file contains presets, update preset dictionary
            if 'presets' in self.custom_config:
                self.presets.update(self.custom_config['presets'])
            
            logger.info(f"Loaded config file: {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}, will use default presets")
    
    def get_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for specified preset.
        
        Args:
            preset_name: Preset name
        
        Returns:
            Configuration dictionary, or None if preset does not exist
        """
        return self.presets.get(preset_name)
    
    def list_presets(self) -> Dict[str, str]:
        """
        List all available presets.
        
        Returns:
            Dictionary mapping preset names to descriptions
        """
        return {name: preset.get('description', '') for name, preset in self.presets.items()}
    
    def save_config(self, config_file: Path, preset_name: Optional[str] = None, 
                    custom_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config_file: Output config file path
            preset_name: Base preset name (optional)
            custom_params: Custom parameters (optional)
        """
        config = {}
        
        if preset_name:
            base_preset = self.get_preset(preset_name)
            if base_preset:
                config = base_preset.copy()
            else:
                logger.warning(f"Preset {preset_name} does not exist, using default configuration")
                config = DEFAULT_PRESETS['default'].copy()
        
        if custom_params:
            config.update(custom_params)
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration saved to: {config_file}")
    
    def apply_preset_to_args(self, preset_name: str, args: Any) -> None:
        """
        Apply preset to command-line arguments object.
        
        Args:
            preset_name: Preset name
            args: argparse.Namespace object
        """
        preset = self.get_preset(preset_name)
        if not preset:
            logger.warning(f"Preset {preset_name} does not exist, skipping application")
            return
        
        # Apply preset parameters to args object
        for key, value in preset.items():
            if key == 'description':
                continue
            if hasattr(args, key):
                setattr(args, key, value)
                logger.debug(f"Applied preset parameter: {key} = {value}")
        
        logger.info(f"Applied preset: {preset_name} ({preset.get('description', '')})")

