"""Entrypoint shim for the `voxtral-server` console script."""
from voxtral.server import main  # noqa: F401

if __name__ == "__main__":
    main()
