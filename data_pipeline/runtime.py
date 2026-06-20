from __future__ import annotations

import os
import sys


def safe_exit_if_requested(enabled: bool) -> None:
    if not enabled:
        return

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
