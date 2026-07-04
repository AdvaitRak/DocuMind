# test_db.py

import os
import time
from dotenv import load_dotenv
import psycopg
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

load_dotenv()

RAW_URL     = os.getenv("DATABASE_URL")
PSYCOPG_URL = RAW_URL.replace("postgresql+psycopg://", "postgresql://")

print(f"URL length : {len(RAW_URL)}")
print(f"URL        : {repr(RAW_URL)}")

# ── Test 1: raw psycopg3 (new connection every time) ─────────────────────────

def test_raw(runs: int = 5):
    print(f"\n── Raw psycopg3 (new connection each run) ───────────")
    times = []
    for i in range(runs):
        start = time.perf_counter()
        try:
            conn = psycopg.connect(PSYCOPG_URL)
            conn.execute("SELECT 1")
            conn.close()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.0f}ms")
        except Exception as e:
            print(f"  Run {i+1}: FAILED — {e}")
    if times:
        print(f"  Average : {sum(times)/len(times):.0f}ms")
        print(f"  Best    : {min(times):.0f}ms")
        print(f"  Worst   : {max(times):.0f}ms")

# ── Test 2: SQLAlchemy connection pool (reuses connections) ───────────────────

def test_pool(runs: int = 5):
    print(f"\n── SQLAlchemy pool (reused connections) ─────────────")
    engine = create_engine(
        RAW_URL,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=300,
    )
    times = []
    for i in range(runs):
        start = time.perf_counter()
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.0f}ms")
        except Exception as e:
            print(f"  Run {i+1}: FAILED — {e}")
    if times:
        print(f"  Average : {sum(times)/len(times):.0f}ms")
        print(f"  Best    : {min(times):.0f}ms")
        print(f"  Worst   : {max(times):.0f}ms")
    engine.dispose()

if __name__ == "__main__":
    test_raw(runs=5)
    test_pool(runs=5)