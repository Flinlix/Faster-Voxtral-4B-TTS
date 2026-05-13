"""Entrypoint shim for the `voxtral-server` console script."""

import sys
import os

# Allow `voxtral-server` to be run from anywhere — add the package root to the
# path so that `import voxtral` resolves regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server import main  # noqa: E402

if __name__ == "__main__":
    main()
