"""
Snowflake integration for event logging and analytics.
"""

import snowflake.connector
from snowflake.connector import DictCursor
import uuid
import logging
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime

from config import (
    SNOWFLAKE_USER,
    SNOWFLAKE_PASSWORD,
    SNOWFLAKE_ACCOUNT,
    SNOWFLAKE_WAREHOUSE,
    SNOWFLAKE_DATABASE,
    SNOWFLAKE_SCHEMA,
    LOG_TO_SNOWFLAKE,
    SESSION_ID
)

logger = logging.getLogger(__name__)


class SnowflakeLogger:
    """Snowflake connection and logging."""

    def __init__(self, enabled: bool = LOG_TO_SNOWFLAKE):
        """
        Initialize Snowflake logger.

        Args:
            enabled: Whether logging is enabled
        """
        self.enabled = enabled and bool(SNOWFLAKE_USER and SNOWFLAKE_ACCOUNT)
        self.conn = None

        if self.enabled:
            try:
                self._connect()
            except Exception as e:
                logger.error(f"Failed to connect to Snowflake: {e}")
                self.enabled = False
        else:
            logger.warning("Snowflake logging is disabled")

    def _connect(self):
        """Establish Snowflake connection."""
        logger.info("Connecting to Snowflake...")

        self.conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA
        )

        logger.info(f"Connected to Snowflake: {SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}")

    def init_db(self):
        """Initialize database schema from SQL files."""
        if not self.enabled:
            logger.warning("Cannot init DB - Snowflake not enabled")
            return

        sql_dir = Path(__file__).parent / "sql"

        # Create tables
        tables_sql = sql_dir / "create_tables.sql"
        if tables_sql.exists():
            logger.info("Creating tables...")
            self._execute_sql_file(tables_sql)

        # Create views
        views_sql = sql_dir / "create_views.sql"
        if views_sql.exists():
            logger.info("Creating views...")
            self._execute_sql_file(views_sql)

        logger.info("Database initialized")

    def _execute_sql_file(self, filepath: Path):
        """Execute SQL from file."""
        with open(filepath, 'r') as f:
            sql = f.read()

        # Split by semicolon and execute each statement
        statements = [s.strip() for s in sql.split(';') if s.strip()]

        cursor = self.conn.cursor()
        try:
            for statement in statements:
                logger.debug(f"Executing: {statement[:100]}...")
                cursor.execute(statement)
            self.conn.commit()
        finally:
            cursor.close()

    def log_event(
        self,
        object_class: str,
        side: str,
        distance_bucket: str,
        ttc_sec: Optional[float] = None,
        confidence: Optional[float] = None,
        instruction: Optional[str] = None,
        source: str = "cv"
    ):
        """
        Log an event to Snowflake.

        Args:
            object_class: Object class or hazard type
            side: left, center, right
            distance_bucket: very_near, near, mid, far
            ttc_sec: Time to collision (if applicable)
            confidence: Detection confidence
            instruction: Spoken instruction
            source: Event source (cv, ocr, rule)
        """
        if not self.enabled:
            return

        event_id = str(uuid.uuid4())
        ts = datetime.utcnow()

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO events (
                    event_id, ts, session_id, object_class, side,
                    distance_bucket, ttc_sec, confidence, instruction, source
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (event_id, ts, SESSION_ID, object_class, side,
                 distance_bucket, ttc_sec, confidence, instruction, source)
            )
            self.conn.commit()
            cursor.close()

            logger.debug(f"Logged event: {object_class} ({side}, {distance_bucket})")

        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def log_ocr_text(
        self,
        snippet: str,
        side: str = "center",
        distance_bucket: str = "unknown"
    ):
        """
        Log OCR text to Snowflake.

        Args:
            snippet: Text snippet
            side: Side of frame
            distance_bucket: Distance estimate
        """
        if not self.enabled:
            return

        event_id = str(uuid.uuid4())
        ts = datetime.utcnow()

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO ocr_text (event_id, ts, snippet, side, distance_bucket)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (event_id, ts, snippet, side, distance_bucket)
            )
            self.conn.commit()
            cursor.close()

            logger.debug(f"Logged OCR: {snippet[:50]}...")

        except Exception as e:
            logger.error(f"Failed to log OCR: {e}")

    def fetch_top_hazards(self, window_minutes: int = 2) -> List[Dict]:
        """
        Fetch top hazards from recent time window.

        Args:
            window_minutes: Time window in minutes

        Returns:
            List of hazard dicts
        """
        if not self.enabled:
            return []

        try:
            cursor = self.conn.cursor(DictCursor)
            cursor.execute(
                """
                SELECT * FROM top_hazards
                WHERE bucket_ts > DATEADD('minute', %s, CURRENT_TIMESTAMP())
                ORDER BY risk_score DESC
                LIMIT 10
                """,
                (-window_minutes,)
            )

            results = cursor.fetchall()
            cursor.close()

            return results

        except Exception as e:
            logger.error(f"Failed to fetch hazards: {e}")
            return []

    def close(self):
        """Close connection."""
        if self.conn:
            self.conn.close()
            logger.info("Snowflake connection closed")


# Global instance
_snowflake_logger: Optional[SnowflakeLogger] = None


def get_snowflake_logger() -> SnowflakeLogger:
    """Get or create global Snowflake logger."""
    global _snowflake_logger
    if _snowflake_logger is None:
        _snowflake_logger = SnowflakeLogger()
    return _snowflake_logger


if __name__ == "__main__":
    # Test Snowflake connection
    logging.basicConfig(level=logging.INFO)

    logger_inst = SnowflakeLogger()

    if logger_inst.enabled:
        print("Testing Snowflake connection...")

        # Init DB
        logger_inst.init_db()

        # Log a test event
        logger_inst.log_event(
            object_class="person",
            side="left",
            distance_bucket="near",
            ttc_sec=2.5,
            confidence=0.9,
            instruction="Person approaching from your left. Pause.",
            source="cv"
        )

        print("Test event logged successfully")

        # Fetch top hazards
        hazards = logger_inst.fetch_top_hazards(window_minutes=10)
        print(f"Recent hazards: {hazards}")

        logger_inst.close()
    else:
        print("Snowflake not configured. Check your .env file.")
