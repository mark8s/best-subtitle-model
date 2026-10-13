"""启动演示服务。访问日志写到项目下的 logs/demo.log。"""

from __future__ import annotations

import copy
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / "logs" / "demo.log"


def main() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    config["handlers"]["access"] = {
        "formatter": "access",
        "class": "logging.handlers.RotatingFileHandler",
        "filename": str(LOG_FILE),
        "maxBytes": 10 * 1024 * 1024,
        "backupCount": 3,
        "encoding": "utf-8",
    }
    print(f"日志 : {LOG_FILE}")
    uvicorn.run("demo_server.app:app", host="0.0.0.0", port=8080, log_config=config)


if __name__ == "__main__":
    main()
