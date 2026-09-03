import logging
import re
from typing import Any


class RobustJSONParser:
    """JSON-repair fallback for a provider with no structured-output mode.

    No provider in this repo uses it today: OpenAI's `responses.parse` gives a
    schema-constrained decode instead, which makes an out-of-schema response
    unrepresentable rather than merely unlikely. Kept for a future provider that lacks
    that guarantee and must be parsed and repaired after the fact.
    """

    def __init__(self, parser: Any):
        self.parser = parser

    def invoke(self, input, config=None):
        return self.parse(input)

    def parse(self, text: str) -> Any:

        try:
            return self.parser.parse(text)
        except Exception as e:
            # Try to clean the string
            cleaned = self._sanitize_json_output(text)
            try:
                return self.parser.parse(cleaned)
            except Exception as inner_e:
                logging.error(f"Sanitized parsing still failed: {inner_e}")
                raise

    def _sanitize_json_output(self, text: str) -> str:

        # Escape unescaped double quotes inside string values
        def escape_inner_quotes(match):
            key, value = match.group(1), match.group(2)
            escaped_value = re.sub(r'(?<!\\)"', r'\\"', value)
            return f'"{key}": "{escaped_value}"'

        # Remove triple backticks and json marker
        text = re.sub(r"```(?:json)?", "", text)
        text = re.sub(r"```", "", text).strip()

        # Fix invalid escape sequences
        text = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)

        # Only simple "key": "value" pattern for now
        # text = re.sub(r'"([^"]+)"\s*:\s*"([^"]*?)"', escape_inner_quotes, text)

        return text
