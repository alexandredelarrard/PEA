#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Package entry point."""
import os
import sys

from cli import main

sys.path.append(os.getcwd())


if __name__ == "__main__":  # pragma: no cover
    main()
