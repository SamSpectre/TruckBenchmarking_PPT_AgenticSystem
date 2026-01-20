"""
Audit Service for E-Powertrain Benchmarking System

Provides enterprise-grade audit trail for human review process.
Tracks review sessions, decisions, and individual field edits.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReviewSession:
    """Represents a human review session."""
    id: Optional[int]
    thread_id: str
    reviewer_id: Optional[str]
    csv_export_path: str
    csv_import_path: Optional[str]
    vehicle_count: int
    quality_score: float
    status: str  # pending, approved, approved_with_edits, rejected
    rejection_reason: Optional[str]
    created_at: str
    completed_at: Optional[str]


@dataclass
class FieldEdit:
    """Represents an individual field edit made during review."""
    id: Optional[int]
    session_id: int
    row_id: int
    vehicle_name: str
    field_name: str
    field_display_name: str
    original_value: str
    new_value: str
    change_type: str  # modified, added, removed
    created_at: str


class AuditService:
    """
    Service for audit trail logging.

    Features:
    - SQLite-based persistent storage
    - Review session tracking
    - Field-level edit logging
    - Query capabilities for reporting
    """

    def __init__(self, db_path: str = "data/audit.db"):
        """
        Initialize audit service.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._ensure_db_dir()
        self._init_database()

    def _ensure_db_dir(self) -> None:
        """Create database directory if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Review sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS review_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    reviewer_id TEXT,
                    csv_export_path TEXT NOT NULL,
                    csv_import_path TEXT,
                    vehicle_count INTEGER NOT NULL,
                    quality_score REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    rejection_reason TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                )
            """)

            # Field edits table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS field_edits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    row_id INTEGER NOT NULL,
                    vehicle_name TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    field_display_name TEXT NOT NULL,
                    original_value TEXT,
                    new_value TEXT,
                    change_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES review_sessions(id)
                )
            """)

            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_thread_id
                ON review_sessions(thread_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_status
                ON review_sessions(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_edits_session_id
                ON field_edits(session_id)
            """)

            logger.info(f"Audit database initialized: {self.db_path}")

    def create_session(
        self,
        thread_id: str,
        csv_export_path: str,
        vehicle_count: int,
        quality_score: float
    ) -> int:
        """
        Create a new review session.

        Args:
            thread_id: Workflow thread ID
            csv_export_path: Path to exported CSV
            vehicle_count: Number of vehicles for review
            quality_score: Quality validation score

        Returns:
            Session ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO review_sessions
                (thread_id, csv_export_path, vehicle_count, quality_score, status, created_at)
                VALUES (?, ?, ?, ?, 'pending', ?)
            """, (
                thread_id,
                csv_export_path,
                vehicle_count,
                quality_score,
                datetime.now().isoformat()
            ))
            session_id = cursor.lastrowid
            logger.info(f"Created review session {session_id} for thread {thread_id}")
            return session_id

    def complete_session(
        self,
        session_id: int,
        status: str,
        reviewer_id: Optional[str] = None,
        csv_import_path: Optional[str] = None,
        rejection_reason: Optional[str] = None
    ) -> None:
        """
        Complete a review session with decision.

        Args:
            session_id: Session ID to complete
            status: Final status (approved, approved_with_edits, rejected)
            reviewer_id: ID of the reviewer
            csv_import_path: Path to imported edited CSV (if any)
            rejection_reason: Reason for rejection (if rejected)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE review_sessions
                SET status = ?,
                    reviewer_id = ?,
                    csv_import_path = ?,
                    rejection_reason = ?,
                    completed_at = ?
                WHERE id = ?
            """, (
                status,
                reviewer_id,
                csv_import_path,
                rejection_reason,
                datetime.now().isoformat(),
                session_id
            ))
            logger.info(f"Completed session {session_id} with status: {status}")

    def log_edits(
        self,
        session_id: int,
        changes: List[Dict[str, Any]]
    ) -> int:
        """
        Log field edits from CSV import.

        Args:
            session_id: Review session ID
            changes: List of change dictionaries from CSV import

        Returns:
            Number of edits logged
        """
        if not changes:
            return 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()

            for change in changes:
                cursor.execute("""
                    INSERT INTO field_edits
                    (session_id, row_id, vehicle_name, field_name, field_display_name,
                     original_value, new_value, change_type, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    change.get("row_id", 0),
                    change.get("vehicle_name", "Unknown"),
                    change.get("field", ""),
                    change.get("field_display_name", ""),
                    self._serialize_value(change.get("original_value")),
                    self._serialize_value(change.get("new_value")),
                    change.get("change_type", "modified"),
                    timestamp
                ))

            logger.info(f"Logged {len(changes)} edits for session {session_id}")
            return len(changes)

    def _serialize_value(self, value: Any) -> str:
        """Serialize a value for storage."""
        if value is None:
            return ""
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return str(value)

    def get_session(self, session_id: int) -> Optional[ReviewSession]:
        """
        Get a review session by ID.

        Args:
            session_id: Session ID

        Returns:
            ReviewSession or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM review_sessions WHERE id = ?
            """, (session_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_session(row)
            return None

    def get_sessions_by_thread(self, thread_id: str) -> List[ReviewSession]:
        """
        Get all review sessions for a thread.

        Args:
            thread_id: Workflow thread ID

        Returns:
            List of ReviewSession objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM review_sessions
                WHERE thread_id = ?
                ORDER BY created_at DESC
            """, (thread_id,))
            return [self._row_to_session(row) for row in cursor.fetchall()]

    def get_pending_sessions(self) -> List[ReviewSession]:
        """
        Get all pending review sessions.

        Returns:
            List of pending ReviewSession objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM review_sessions
                WHERE status = 'pending'
                ORDER BY created_at ASC
            """)
            return [self._row_to_session(row) for row in cursor.fetchall()]

    def get_session_edits(self, session_id: int) -> List[FieldEdit]:
        """
        Get all field edits for a session.

        Args:
            session_id: Review session ID

        Returns:
            List of FieldEdit objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM field_edits
                WHERE session_id = ?
                ORDER BY row_id, field_name
            """, (session_id,))
            return [self._row_to_edit(row) for row in cursor.fetchall()]

    def get_recent_sessions(self, limit: int = 10) -> List[ReviewSession]:
        """
        Get most recent review sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of ReviewSession objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM review_sessions
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            return [self._row_to_session(row) for row in cursor.fetchall()]

    def get_audit_summary(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics for audit trail.

        Args:
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)

        Returns:
            Summary dictionary
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build date filter
            date_filter = ""
            params = []
            if start_date:
                date_filter += " AND created_at >= ?"
                params.append(start_date)
            if end_date:
                date_filter += " AND created_at <= ?"
                params.append(end_date)

            # Session counts by status
            cursor.execute(f"""
                SELECT status, COUNT(*) as count
                FROM review_sessions
                WHERE 1=1 {date_filter}
                GROUP BY status
            """, params)
            status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}

            # Total edits
            cursor.execute(f"""
                SELECT COUNT(*) as count
                FROM field_edits fe
                JOIN review_sessions rs ON fe.session_id = rs.id
                WHERE 1=1 {date_filter}
            """, params)
            total_edits = cursor.fetchone()["count"]

            # Edit counts by type
            cursor.execute(f"""
                SELECT change_type, COUNT(*) as count
                FROM field_edits fe
                JOIN review_sessions rs ON fe.session_id = rs.id
                WHERE 1=1 {date_filter}
                GROUP BY change_type
            """, params)
            edit_counts = {row["change_type"]: row["count"] for row in cursor.fetchall()}

            # Most edited fields
            cursor.execute(f"""
                SELECT field_display_name, COUNT(*) as count
                FROM field_edits fe
                JOIN review_sessions rs ON fe.session_id = rs.id
                WHERE 1=1 {date_filter}
                GROUP BY field_display_name
                ORDER BY count DESC
                LIMIT 5
            """, params)
            top_edited_fields = [
                {"field": row["field_display_name"], "count": row["count"]}
                for row in cursor.fetchall()
            ]

            return {
                "total_sessions": sum(status_counts.values()),
                "sessions_by_status": status_counts,
                "total_edits": total_edits,
                "edits_by_type": edit_counts,
                "top_edited_fields": top_edited_fields,
                "date_range": {
                    "start": start_date,
                    "end": end_date,
                },
            }

    def _row_to_session(self, row: sqlite3.Row) -> ReviewSession:
        """Convert database row to ReviewSession object."""
        return ReviewSession(
            id=row["id"],
            thread_id=row["thread_id"],
            reviewer_id=row["reviewer_id"],
            csv_export_path=row["csv_export_path"],
            csv_import_path=row["csv_import_path"],
            vehicle_count=row["vehicle_count"],
            quality_score=row["quality_score"],
            status=row["status"],
            rejection_reason=row["rejection_reason"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )

    def _row_to_edit(self, row: sqlite3.Row) -> FieldEdit:
        """Convert database row to FieldEdit object."""
        return FieldEdit(
            id=row["id"],
            session_id=row["session_id"],
            row_id=row["row_id"],
            vehicle_name=row["vehicle_name"],
            field_name=row["field_name"],
            field_display_name=row["field_display_name"],
            original_value=row["original_value"],
            new_value=row["new_value"],
            change_type=row["change_type"],
            created_at=row["created_at"],
        )

    def export_session_report(self, session_id: int) -> Dict[str, Any]:
        """
        Export complete report for a review session.

        Args:
            session_id: Review session ID

        Returns:
            Complete session report dictionary
        """
        session = self.get_session(session_id)
        if not session:
            return {"error": f"Session {session_id} not found"}

        edits = self.get_session_edits(session_id)

        return {
            "session": asdict(session),
            "edits": [asdict(edit) for edit in edits],
            "summary": {
                "total_edits": len(edits),
                "edits_by_type": self._count_edits_by_type(edits),
                "edits_by_vehicle": self._count_edits_by_vehicle(edits),
            },
            "generated_at": datetime.now().isoformat(),
        }

    def _count_edits_by_type(self, edits: List[FieldEdit]) -> Dict[str, int]:
        """Count edits by change type."""
        counts = {}
        for edit in edits:
            counts[edit.change_type] = counts.get(edit.change_type, 0) + 1
        return counts

    def _count_edits_by_vehicle(self, edits: List[FieldEdit]) -> Dict[str, int]:
        """Count edits by vehicle."""
        counts = {}
        for edit in edits:
            counts[edit.vehicle_name] = counts.get(edit.vehicle_name, 0) + 1
        return counts


# Module-level instance
_default_service: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    """Get the default audit service instance."""
    global _default_service
    if _default_service is None:
        _default_service = AuditService()
    return _default_service


if __name__ == "__main__":
    # Test the audit service
    print("=" * 60)
    print("AUDIT SERVICE TEST")
    print("=" * 60)

    # Use test database
    service = AuditService("data/audit_test.db")

    # Create a session
    session_id = service.create_session(
        thread_id="test_thread_001",
        csv_export_path="/tmp/test_export.csv",
        vehicle_count=5,
        quality_score=0.85
    )
    print(f"\nCreated session: {session_id}")

    # Log some edits
    test_changes = [
        {
            "row_id": 1,
            "vehicle_name": "MAN eTGX",
            "field": "battery_capacity_kwh",
            "field_display_name": "Battery (kWh)",
            "original_value": 480,
            "new_value": 500,
            "change_type": "modified",
        },
        {
            "row_id": 2,
            "vehicle_name": "MAN eTGS",
            "field": "range_km",
            "field_display_name": "Range (km)",
            "original_value": None,
            "new_value": 650,
            "change_type": "added",
        },
    ]
    edits_logged = service.log_edits(session_id, test_changes)
    print(f"Logged {edits_logged} edits")

    # Complete session
    service.complete_session(
        session_id=session_id,
        status="approved_with_edits",
        reviewer_id="test_user",
        csv_import_path="/tmp/test_import.csv"
    )
    print("Session completed")

    # Get session details
    session = service.get_session(session_id)
    print(f"\nSession status: {session.status}")
    print(f"Reviewer: {session.reviewer_id}")

    # Get edits
    edits = service.get_session_edits(session_id)
    print(f"\nEdits ({len(edits)}):")
    for edit in edits:
        print(f"  {edit.vehicle_name}: {edit.field_display_name} = {edit.original_value} -> {edit.new_value}")

    # Get summary
    summary = service.get_audit_summary()
    print(f"\nAudit Summary:")
    print(f"  Total sessions: {summary['total_sessions']}")
    print(f"  Total edits: {summary['total_edits']}")

    # Export report
    report = service.export_session_report(session_id)
    print(f"\nSession report exported with {report['summary']['total_edits']} edits")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
