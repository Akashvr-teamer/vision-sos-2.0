# database/database.py
# database/database.py

import sqlite3
import os


DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "events.db"
)


def get_connection():
    """
    Create a new SQLite connection.

    A new connection is created for every operation
    to avoid SQLite thread-related problems.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Create the events table if it does not already exist.
    """

    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            confidence REAL,
            severity INTEGER,
            severity_label TEXT,
            motionless_sec REAL,
            fall_velocity REAL,
            timestamp TEXT,
            snapshot_path TEXT,
            alert_sent INTEGER DEFAULT 0,
            category TEXT
        )
    """)

    # Try to add category column if table exists from before
    try:
        conn.execute("ALTER TABLE events ADD COLUMN category TEXT")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

    print("[DB] Database initialized")


def save_event(event, severity, severity_label, snapshot_path, alert_sent):
    """
    Save an emergency event to the database.
    """

    conn = get_connection()

    cursor = conn.execute("""
        INSERT INTO events (
            event_type,
            confidence,
            severity,
            severity_label,
            motionless_sec,
            fall_velocity,
            timestamp,
            snapshot_path,
            alert_sent,
            category
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event["type"],
        event["confidence"],
        severity,
        severity_label,
        event["motionless_sec"],
        event["fall_velocity"],
        event["timestamp"],
        snapshot_path,
        int(alert_sent),
        event.get("category", event["type"].replace("_", " ").title())
    ))

    event_id = cursor.lastrowid

    conn.commit()
    conn.close()

    print(f"[DB] Saved event #{event_id}: {event['type']}")

    return event_id


def get_events(limit=20):
    """
    Get the most recent emergency events.
    """

    conn = get_connection()

    rows = conn.execute("""
        SELECT *
        FROM events
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()

    conn.close()

    return [dict(row) for row in rows]


def update_event_category(event_id, category):
    """
    Update the category of an event.
    """
    conn = get_connection()
    conn.execute(
        "UPDATE events SET category = ? WHERE id = ?",
        (category, event_id)
    )
    conn.commit()
    conn.close()
    return True