# -*- coding: utf-8 -*-
"""
Some utils for string manipulations
"""
import math
import re
from typing import List, Union


def camel_to_snake(s: str) -> str:
    """
    Transforms a camel type cased string (e.g. "MyClass") to snake cased type (e.g. "my_class")
    Args:
        s: the string to be converted
    Returns:
        s_conv: converted string
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
