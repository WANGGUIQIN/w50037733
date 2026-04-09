"""Disk space monitoring for streaming pipeline."""
import shutil


def check_disk(
    path: str = "/home/edge",
    warn_gb: float = 100,
    critical_gb: float = 50,
) -> str:
    """Returns 'ok', 'warn', or 'critical'."""
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < critical_gb:
        return "critical"
    if free_gb < warn_gb:
        return "warn"
    return "ok"


def get_free_gb(path: str = "/home/edge") -> float:
    return shutil.disk_usage(path).free / 1e9
