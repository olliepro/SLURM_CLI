from __future__ import annotations

import sys

from slurm_cli.cli import main


if __name__ == "__main__":
    raise SystemExit(main(argv=sys.argv[1:]))
