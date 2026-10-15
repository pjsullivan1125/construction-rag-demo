"""Load the fictional RFI log and submittal register into a SQLite file for SQL practice.

Run from the project root:  python sql/make_db.py
Then:                       sqlite3 construction.db   (or run queries from Python)
"""
import csv
import sqlite3
from pathlib import Path

root = Path(__file__).resolve().parent.parent
db_path = root / "construction.db"
if db_path.exists():
    db_path.unlink()

con = sqlite3.connect(db_path)

def load(table, csv_file):
    with open(root / "data" / csv_file, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0].keys())
    con.execute(f"CREATE TABLE {table} ({', '.join(c + ' TEXT' for c in cols)})")
    con.executemany(
        f"INSERT INTO {table} VALUES ({', '.join('?' for _ in cols)})",
        [[(r[c] if r[c] != '' else None) for c in cols] for r in rows],
    )
    print(f"Loaded {len(rows)} rows into {table}")

load("rfis", "rfi_log.csv")
load("submittals", "submittal_register.csv")
con.commit()
con.close()
