"""Eval-dataset loading for v2 (thin re-export of the v1 reader).

v1's `load_eval_games` already parses this harness's eval JSONL format, and the
format has not changed, so v2 reuses it rather than maintaining a second parser.
The import is read-only: v1 files are never modified.
"""

from tasks.alfworld.v1.rollout import load_eval_games

__all__ = ["load_eval_games"]
