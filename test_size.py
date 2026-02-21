import importlib
import os
import sys
import tracemalloc

# Ensure src/ modules are importable when running from repo root.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

def _format_bytes(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.2f} {unit}"
        num //= 1024
    return f"{num:.2f} TB"


def measure_import_bytes(module_name: str) -> int:
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    importlib.import_module(module_name)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    return sum(stat.size_diff for stat in after.compare_to(before, "filename"))


def measure_object_bytes(obj) -> int:
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(sys.getsizeof(item) for item in obj)
    return size


def main() -> None:
    modules = ["lookups_v2", "lookups"]
    for name in modules:
        import_bytes = measure_import_bytes(name)
        #module = importlib.import_module(name)
        print(f"{name}: import_bytes: {import_bytes}")


if __name__ == "__main__":
    main()

