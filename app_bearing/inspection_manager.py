# inspection_manager.py
import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from config import DAILY_TARGET, SHIFT_HOURS


class InspectionManager:
    def __init__(self, db_path="inspection_qc.db", legacy_json_file=None):
        self.db_path = str(db_path)
        self.legacy_json_file = legacy_json_file
        self._lock = threading.Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._migrate_legacy_json()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS qc_inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    shift TEXT NOT NULL,
                    image_name TEXT,
                    measurement_json TEXT,
                    status TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_qc_inspections_timestamp "
                "ON qc_inspections(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_qc_inspections_class "
                "ON qc_inspections(class)"
            )

    def _migrate_legacy_json(self):
        if not self.legacy_json_file or not os.path.exists(self.legacy_json_file):
            return

        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM qc_inspections").fetchone()[0]
            if count:
                return

        try:
            with open(self.legacy_json_file, "r", encoding="utf-8") as f:
                legacy_logs = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(legacy_logs, list):
            return

        with self._lock:
            with self._connect() as conn:
                for entry in legacy_logs:
                    if not isinstance(entry, dict):
                        continue
                    self._insert_entry(conn, entry)

    def _insert_entry(self, conn, entry):
        measurement = entry.get("measurement")
        measurement_json = json.dumps(measurement) if measurement is not None else None
        return conn.execute(
            """
            INSERT INTO qc_inspections (
                timestamp, class, confidence, shift, image_name, measurement_json, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.get("timestamp") or datetime.now().isoformat(),
                entry.get("class") or "unknown",
                float(entry.get("confidence") or 0),
                entry.get("shift") or self.get_current_shift(),
                entry.get("image_name") or "upload",
                measurement_json,
                entry.get("status") or (measurement or {}).get("status"),
            ),
        )

    def _row_to_dict(self, row):
        measurement = None
        if row["measurement_json"]:
            try:
                measurement = json.loads(row["measurement_json"])
            except json.JSONDecodeError:
                measurement = None

        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "class": row["class"],
            "confidence": row["confidence"],
            "shift": row["shift"],
            "image_name": row["image_name"],
            "measurement": measurement,
            "status": row["status"],
        }

    def _size_bucket(self, class_name):
        normalized = str(class_name or "").strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"small", "small_bearing", "688z"}:
            return "small"
        if normalized in {"medium", "medium_bearing", "608z"}:
            return "medium"
        if normalized in {"large", "large_bearing", "big_bearings", "big_bearing", "6301z"}:
            return "large"
        if normalized == "no_bearing":
            return "no_bearing"
        return normalized

    def _class_filter_values(self, class_filter):
        aliases = {
            "small": ["small", "small_bearing", "688Z"],
            "medium": ["medium", "medium_bearing", "608Z"],
            "large": ["large", "large_bearing", "big_bearings", "big_bearing", "6301Z"],
            "no_bearing": ["no_bearing"],
        }
        return aliases.get(str(class_filter or "").lower(), [class_filter])

    def _bearing_type(self, class_name, measurement=None):
        if measurement and measurement.get("class"):
            return measurement["class"]
        bucket = self._size_bucket(class_name)
        if bucket == "small":
            return "688Z"
        if bucket == "medium":
            return "608Z"
        if bucket == "large":
            return "6301Z"
        if bucket == "no_bearing":
            return "No Bearing"
        return "Unknown"

    @property
    def logs(self):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM qc_inspections ORDER BY timestamp ASC, id ASC"
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def add(self, class_name, confidence, image_name="upload", measurement=None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "class": class_name,
            "confidence": confidence,
            "shift": self.get_current_shift(),
            "image_name": image_name,
            "measurement": measurement,
            "status": measurement.get("status") if measurement else None,
        }
        with self._lock:
            with self._connect() as conn:
                cursor = self._insert_entry(conn, entry)
                entry["id"] = cursor.lastrowid
        return entry

    def get_current_shift(self):
        hour = datetime.now().hour
        for shift, (start, end) in SHIFT_HOURS.items():
            if start <= end:
                if start <= hour < end:
                    return shift
            else:
                if hour >= start or hour < end:
                    return shift
        return "Morning"

    def get_statistics(self, scope="today"):
        if scope == "all":
            start = "0000-01-01T00:00:00"
            end = "9999-12-31T23:59:59.999999"
            label = "All QC Data"
        else:
            today = datetime.now().date().isoformat()
            start = f"{today}T00:00:00"
            end = f"{today}T23:59:59.999999"
            label = "Today"

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT class, status, COUNT(*) AS count
                FROM qc_inspections
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY class, status
                """,
                (start, end),
            ).fetchall()
            recent_rows = conn.execute(
                """
                SELECT * FROM qc_inspections
                ORDER BY timestamp DESC, id DESC
                LIMIT 20
                """
            ).fetchall()

        total = sum(row["count"] for row in rows)
        small = sum(row["count"] for row in rows if self._size_bucket(row["class"]) == "small")
        medium = sum(row["count"] for row in rows if self._size_bucket(row["class"]) == "medium")
        large = sum(row["count"] for row in rows if self._size_bucket(row["class"]) == "large")
        no_bearing = sum(row["count"] for row in rows if self._size_bucket(row["class"]) == "no_bearing")
        reject = sum(row["count"] for row in rows if row["status"] == "REJECT")
        ok = sum(row["count"] for row in rows if row["status"] == "PASS")
        warning = 0
        unknown = max(0, total - ok - reject - warning)

        return {
            "scope": scope,
            "label": label,
            "total": total,
            "small": small,
            "medium": medium,
            "large": large,
            "no_bearing": no_bearing,
            "ok": ok,
            "warning": warning,
            "reject": reject,
            "unknown": unknown,
            "defect_rate": (reject / total * 100) if total > 0 else 0,
            "target_achievement": (total / DAILY_TARGET * 100) if DAILY_TARGET > 0 else 0,
            "shift": self.get_current_shift(),
            "recent_logs": [self._row_to_dict(row) for row in recent_rows],
        }

    def get_logs(self, class_filter=None, date_from=None, date_to=None, limit=100):
        where = []
        params = []
        if class_filter:
            class_values = self._class_filter_values(class_filter)
            placeholders = ", ".join("?" for _ in class_values)
            where.append(f"class IN ({placeholders})")
            params.extend(class_values)
        if date_from:
            where.append("timestamp >= ?")
            params.append(f"{date_from}T00:00:00" if len(date_from) == 10 else date_from)
        if date_to:
            where.append("timestamp <= ?")
            params.append(f"{date_to}T23:59:59.999999" if len(date_to) == 10 else date_to)

        try:
            limit = max(1, min(int(limit), 1000))
        except (TypeError, ValueError):
            limit = 100

        query = "SELECT * FROM qc_inspections"
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_report(self, date_from=None, date_to=None):
        if not date_from and not date_to:
            start = "0000-01-01T00:00:00"
            end = "9999-12-31T23:59:59.999999"
        else:
            if not date_from:
                date_from = date_to
            if not date_to:
                date_to = date_from
            start = f"{date_from}T00:00:00" if len(date_from) == 10 else date_from
            end = f"{date_to}T23:59:59.999999" if len(date_to) == 10 else date_to

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    substr(timestamp, 1, 10) AS inspection_date,
                    class,
                    status,
                    measurement_json,
                    COUNT(*) AS count
                FROM qc_inspections
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY inspection_date, class, status, measurement_json
                ORDER BY inspection_date ASC
                """,
                (start, end),
            ).fetchall()

        daily = {}
        bearing_distribution = {"688Z": 0, "608Z": 0, "6301Z": 0, "Unknown": 0}
        status_counts = {"PASS": 0, "REJECT": 0, "UNKNOWN": 0}

        for row in rows:
            date_key = row["inspection_date"]
            bucket = self._size_bucket(row["class"])
            measurement = None
            if row["measurement_json"]:
                try:
                    measurement = json.loads(row["measurement_json"])
                except json.JSONDecodeError:
                    measurement = None
            bearing_type = self._bearing_type(row["class"], measurement)
            status = row["status"] or "UNKNOWN"
            count = row["count"]

            if date_key not in daily:
                daily[date_key] = {"date": date_key, "total": 0, "pass": 0, "reject": 0}

            daily[date_key]["total"] += count
            if status == "PASS":
                daily[date_key]["pass"] += count
                status_counts["PASS"] += count
            elif status == "REJECT":
                daily[date_key]["reject"] += count
                status_counts["REJECT"] += count
            else:
                status_counts["UNKNOWN"] += count

            if bearing_type not in bearing_distribution:
                bearing_type = "Unknown"
            bearing_distribution[bearing_type] += count

        daily_rows = []
        for item in daily.values():
            item["defect_rate"] = (item["reject"] / item["total"] * 100) if item["total"] else 0
            daily_rows.append(item)

        return {
            "date_from": date_from,
            "date_to": date_to,
            "daily": daily_rows,
            "bearing_distribution": bearing_distribution,
            "status_counts": status_counts,
            "total": sum(item["total"] for item in daily_rows),
        }
