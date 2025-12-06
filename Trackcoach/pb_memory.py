"""PB lap memory system: track across sessions and update best lap time."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class PBMemory:
    """PB lap memory manager."""
    
    def __init__(self, memory_file: Path):
        """
        Initialize PB memory manager.
        
        Args:
            memory_file: Memory file path (JSON format)
        """
        self.memory_file = Path(memory_file)
        self.memory: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """Load PB memory from file."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.memory = json.load(f)
                logger.info(f"Loaded PB memory: {self.memory_file}")
            except Exception as e:
                logger.warning(f"Failed to load PB memory: {e}, will create new memory")
                self.memory = {
                    'current_pb': None,
                    'pb_history': [],
                    'last_updated': None
                }
        else:
            self.memory = {
                'current_pb': None,
                'pb_history': [],
                'last_updated': None
            }
            logger.info("Creating new PB memory")
    
    def save(self) -> None:
        """Save PB memory to file."""
        self.memory['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)
        
        logger.info(f"PB memory saved: {self.memory_file}")
    
    def get_current_pb(self) -> Optional[Dict[str, Any]]:
        """
        Get current PB lap information.
        
        Returns:
            PB lap information dictionary containing: lap, laptime_s, session_name, session_date, session_dir
        """
        return self.memory.get('current_pb')
    
    def update_pb(
        self,
        lap: int,
        laptime_s: float,
        session_name: str,
        session_date: Optional[str] = None,
        session_dir: Optional[Path] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Update PB lap (if new lap time is faster).
        
        Args:
            lap: Lap number
            laptime_s: Lap time (seconds)
            session_name: Session name
            session_date: Session date (optional)
            session_dir: Session directory path (optional)
        
        Returns:
            (whether PB was updated, old PB information)
        """
        current_pb = self.get_current_pb()
        old_pb = current_pb.copy() if current_pb else None
        
        # If no current PB, or new lap time is faster, then update
        should_update = False
        if current_pb is None:
            should_update = True
            logger.info(f"First PB record: Lap {lap}, {laptime_s:.2f}s")
        elif laptime_s < current_pb['laptime_s']:
            should_update = True
            improvement = current_pb['laptime_s'] - laptime_s
            logger.info(
                f"PB updated! New PB: Lap {lap}, {laptime_s:.2f}s "
                f"(faster than old PB by {improvement:.2f}s, old PB: {current_pb['laptime_s']:.2f}s)"
            )
        else:
            slower_by = laptime_s - current_pb['laptime_s']
            logger.info(
                f"Current lap time {laptime_s:.2f}s did not exceed PB {current_pb['laptime_s']:.2f}s "
                f"(slower by {slower_by:.2f}s), keeping historical PB"
            )
        
        if should_update:
            new_pb = {
                'lap': int(lap),
                'laptime_s': float(laptime_s),
                'session_name': session_name,
                'session_date': session_date or datetime.now().strftime('%Y-%m-%d'),
                'session_dir': str(session_dir) if session_dir else None,
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.memory['current_pb'] = new_pb
            
            # Add to history record
            if old_pb:
                self.memory['pb_history'].append({
                    'old_pb': old_pb,
                    'new_pb': new_pb,
                    'improvement': float(current_pb['laptime_s'] - laptime_s),
                    'updated_at': new_pb['updated_at']
                })
            
            self.save()
            return True, old_pb
        
        return False, None
    
    def get_pb_history(self) -> list:
        """
        Get PB update history.
        
        Returns:
            PB update history list
        """
        return self.memory.get('pb_history', [])
    
    def get_recommended_pb(self) -> Optional[Dict[str, Any]]:
        """
        Get recommended PB lap (for analysis benchmark).
        
        If current session PB is faster, return current PB; otherwise return historical PB.
        
        Returns:
            Recommended PB lap information
        """
        return self.get_current_pb()
    
    def format_pb_info(self, pb: Optional[Dict[str, Any]] = None) -> str:
        """
        Format PB information as string.
        
        Args:
            pb: PB information dictionary, if None then use current PB
        
        Returns:
            Formatted string
        """
        if pb is None:
            pb = self.get_current_pb()
        
        if pb is None:
            return "No PB record"
        
        return (
            f"Lap {pb['lap']}, {pb['laptime_s']:.2f}s "
            f"(Session: {pb['session_name']}, {pb['session_date']})"
        )

