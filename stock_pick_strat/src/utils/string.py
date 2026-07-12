import re


def camel_to_snake(x: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", x).lower()
