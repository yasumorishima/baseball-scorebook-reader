"""Scorebook photo -> plate appearances -> game result and batting stats.

The pipeline:
  1. grid   : find the scoring grid on the photo (cv/grid_detect.py, cv/cell_diff.py)
  2. crops  : cut one image per inning column
  3. read   : Claude reads each column (two prompts: per-cell facts, and base corners)
  4. game   : assemble plate appearances, split columns into innings
  5. stats  : line score and per-slot batting lines

Readers are swappable: ApiReader calls the Anthropic API; ReplayReader serves
replies stored on disk, so the whole chain runs without an API key.
"""
