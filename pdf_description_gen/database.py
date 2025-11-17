#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database operations for materials processing.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional

from .types import MaterialRow


class DatabaseManager:
    """Manages database operations for materials."""
    
    def __init__(self, db_path: str):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
    
    def __enter__(self) -> 'DatabaseManager':
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def connect(self) -> None:
        """Connect to the database."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found at {self.db_path}")
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()
        self._ensure_schema()
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def _ensure_schema(self) -> None:
        """Ensure database schema has required columns (idempotent migration)."""
        if not self.cursor:
            return
        
        try:
            self.cursor.execute('PRAGMA table_info(materials)')
            cols = [r[1] for r in self.cursor.fetchall()]
            
            if 'needed' not in cols:
                self.cursor.execute('ALTER TABLE materials ADD COLUMN needed INTEGER')
            if 'key_concept' not in cols:
                self.cursor.execute('ALTER TABLE materials ADD COLUMN key_concept TEXT')
            
            if self.conn:
                self.conn.commit()
        except Exception:
            pass
    
    def fetch_all_materials(self) -> List[MaterialRow]:
        """
        Fetch all materials from database.
        
        Returns:
            List of MaterialRow objects
        """
        if not self.cursor:
            raise RuntimeError("Database not connected")
        
        self.cursor.execute(
            'SELECT id, original_filename, current_filename, status, description, needed, key_concept '
            'FROM materials'
        )
        
        return [MaterialRow.from_db_row(row) for row in self.cursor.fetchall()]
    
    def update_material(
        self,
        material_id: int,
        status: str,
        description: str,
        needed: Optional[int],
        key_concept: str
    ) -> None:
        """
        Update a material row in the database.
        
        Args:
            material_id: ID of the material to update
            status: Processing status
            description: Generated description
            needed: Whether the material is needed (0/1/None)
            key_concept: Key concept(s)
        """
        if not self.cursor or not self.conn:
            raise RuntimeError("Database not connected")
        
        # Normalize needed value
        if isinstance(needed, bool):
            needed = 1 if needed else 0
        
        self.cursor.execute(
            '''UPDATE materials 
               SET status = ?, description = ?, needed = ?, key_concept = ?
               WHERE id = ?''',
            (status, description, needed, key_concept, material_id)
        )
    
    def update_materials_batch(self, rows: List[MaterialRow]) -> None:
        """
        Update multiple material rows in the database.
        
        Args:
            rows: List of MaterialRow objects to update
        """
        for row in rows:
            self.update_material(
                row.id,
                row.status,
                row.description,
                row.needed,
                row.key_concept
            )
    
    def commit(self) -> None:
        """Commit pending transactions."""
        if self.conn:
            self.conn.commit()

