#!/usr/bin/env python
"""
Setup script for Movie Data Analysis Platform.

This file exists for backwards compatibility with older build tools.
All configuration is now in pyproject.toml (PEP 517/518).

For editable installs:
    pip install -e .

For development installs:
    pip install -e ".[dev]"

For all dependencies:
    pip install -e ".[all]"
"""

from setuptools import setup

# All configuration is in pyproject.toml
# This file exists for backwards compatibility only
if __name__ == "__main__":
    setup()
