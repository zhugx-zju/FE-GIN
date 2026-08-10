"""Evaluate the configured FNO checkpoint on the shared fixed test sets."""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fno.config import get_config
from fno.test import test_fno


test_fno(get_config("custom"))
