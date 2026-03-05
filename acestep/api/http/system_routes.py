"""HTTP routes for system metrics and log streaming."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI


def register_system_routes(
    app: FastAPI,
    *,
    log_buffer: Any,
) -> None:
    """Register system monitoring endpoints."""

    @app.get("/v1/system/metrics")
    async def system_metrics():
        """Return live system metrics: GPU VRAM, system RAM, CPU."""
        import torch

        gpu = {}
        if torch.cuda.is_available():
            try:
                idx = torch.cuda.current_device()
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
                allocated_bytes = torch.cuda.memory_allocated(idx)
                reserved_bytes = torch.cuda.memory_reserved(idx)
                gpu = {
                    "name": torch.cuda.get_device_name(idx),
                    "allocated_gb": round(allocated_bytes / (1024**3), 2),
                    "reserved_gb": round(reserved_bytes / (1024**3), 2),
                    "free_gb": round(free_bytes / (1024**3), 2),
                    "total_gb": round(total_bytes / (1024**3), 2),
                }
            except Exception:
                gpu = {"error": "Failed to read GPU metrics"}
        else:
            gpu = {"error": "No CUDA device available"}

        ram = {}
        cpu = {}
        try:
            import psutil
            mem = psutil.virtual_memory()
            ram = {
                "used_gb": round(mem.used / (1024**3), 2),
                "total_gb": round(mem.total / (1024**3), 2),
                "percent": mem.percent,
            }
            cpu = {
                "percent": psutil.cpu_percent(interval=0),
                "count": psutil.cpu_count(logical=True),
            }
        except ImportError:
            ram = {"error": "psutil not installed"}
            cpu = {"error": "psutil not installed"}
        except Exception as e:
            ram = {"error": str(e)}
            cpu = {"error": str(e)}

        return {"gpu": gpu, "ram": ram, "cpu": cpu}

    @app.get("/v1/system/logs")
    async def system_logs(after: int = -1):
        """Return log lines after the given cursor for efficient polling."""
        lines, current_cursor = log_buffer.get_lines_after(after)
        return {
            "lines": [line for _, line in lines],
            "cursor": current_cursor,
        }
