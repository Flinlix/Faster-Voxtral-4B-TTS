"""Repo-root shim that exposes ``voxtral.server`` as ``server.py``.

Lets ``python server.py`` continue to work after the implementation moved
into the ``voxtral`` package for proper distribution. All real logic lives
in :mod:`voxtral.server`.
"""

from voxtral.server import main

if __name__ == "__main__":
    main()
