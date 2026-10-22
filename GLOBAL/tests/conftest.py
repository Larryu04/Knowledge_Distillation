import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_platform(tmp_path, monkeypatch):
    """Every test gets its own data dir + DB, and the offline fake teacher."""
    monkeypatch.setenv("GLOBAL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GLOBAL_TEACHER_BACKEND", "fake")
    monkeypatch.setenv("GLOBAL_TOOLCHAIN_DIR", str(tmp_path / "no-tools"))
    monkeypatch.setenv("GLOBAL_GEN_PACING_S", "0")
    monkeypatch.delenv("GLOBAL_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("GLOBAL_SECRET_KEY", raising=False)
    # Stage subprocesses must find the package and inherit the env above.
    monkeypatch.setenv("PYTHONPATH", str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    from corecs_global import config, db

    config.get_settings.cache_clear()
    db.init_db()
    yield
    config.get_settings.cache_clear()


SYLLABUS_A = """Course: Data Structures
Weekly units:
1. Arrays and dynamic arrays, amortized analysis of append
2. Singly and doubly linked lists, insertion and deletion
3. Stacks and queues, array and list implementations
4. Hash tables, chaining vs open addressing, load factor
5. Binary search trees, insertion, deletion, traversal orders
6. Heaps and priority queues, heapify, heap sort
7. Balanced trees: AVL rotations
8. Graph representations: adjacency list vs matrix
Out of scope: dynamic programming (taught in Algorithms).
"""

SYLLABUS_B = """Course: Operating Systems
1. Processes and threads
2. CPU scheduling: FCFS, SJF, Round Robin
3. Synchronization: mutexes, semaphores, monitors
4. Deadlocks: conditions, prevention, avoidance (Banker's algorithm)
5. Memory management: paging, segmentation, TLB
6. Virtual memory and page replacement (FIFO, LRU, Clock)
7. File systems: inodes, directories, allocation methods
""" * 2
