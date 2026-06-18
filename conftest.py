"""Pytest collection config.

Excludes the ``archive/`` tree (legacy fracture / Nine-Circles code kept for history,
not part of the active contact-mechanics suite). This works on all pytest versions,
including the local pytest 5.x that does not read ``[tool.pytest.ini_options]`` in
pyproject.toml.
"""
collect_ignore = ["archive"]
collect_ignore_glob = ["archive/*"]
