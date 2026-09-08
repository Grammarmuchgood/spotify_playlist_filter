"""Process one playlist from the command line, without going through the
web app - useful for backfilling, re-running after a pipeline fix, or
debugging a specific playlist directly.

Empty since this project's very first commit - the orchestration it
wraps (pipeline.process_playlist.process_playlist) didn't exist until
multi-user playlist support was added, and every use since then went
through the web app's own POST /playlists/{id}/process instead. This is
the script-level entry point for the same underlying function, for when
a browser isn't the right tool.

Usage (run from the project root):
    .venv/bin/python3 scripts/run_pipeline.py <spotify_user_id> <playlist_id>

The user must have already logged in at least once through the real app
- this reads their saved token from backend/data/users/{user_id}/, it
doesn't perform its own login.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Every module under backend/ imports as if backend/ itself were the
# project root (e.g. `from db.database import ...`, not
# `from backend.db.database import ...`) - matching how the app is
# actually run in production (uvicorn started from inside backend/) and
# how pytest.ini points pythonpath at it for the test suite. Adding it
# here explicitly is what lets this script work when invoked from the
# project root instead of from inside backend/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from pipeline.process_playlist import process_playlist  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("user_id", help="The Spotify user ID whose playlist this is (must have already logged in once via the app)")
    parser.add_argument("playlist_id", help="The Spotify playlist ID to process")
    args = parser.parse_args()

    print(f"Processing playlist {args.playlist_id} for user {args.user_id}...")
    process_playlist(args.user_id, args.playlist_id)
    print("Done.")


if __name__ == "__main__":
    main()
