from typing import Any
import click
from click.core import Context, Parameter
import pandas as pd
from click import ParamType

from src.constants.variables import DATE_FORMAT


class SpecialHelpOrder(click.Group):

    def __init__(self, *args, **kwargs):
        self.help_priorities = {}
        super(SpecialHelpOrder, self).__init__(*args, **kwargs)

    def get_help(self, ctx):
        self.list_commands = self.list_commands_for_help
        print(self.list_commands)
        return super(SpecialHelpOrder, self).get_help(ctx)

    def list_commands_for_help(self, ctx):
        commands = super(SpecialHelpOrder, self).list_commands(ctx)
        return (
            c[1]
            for c in sorted(
                (self.help_priorities.get(command, 99), command) for command in commands
            )
        )

    def command(self, *args, **kwargs):
        help_priority = kwargs.pop("help_priority", 99)
        help_priorities = self.help_priorities

        def decorator(f):
            cmd = super(SpecialHelpOrder, self).command(*args, **kwargs)(f)
            help_priorities[cmd.name] = help_priority
            return cmd

        return decorator


class CLITimestamp(ParamType):
    name = "timestamp"

    def convert(self, value: Any, param: Parameter | None, ctx: Context | None) -> Any:
        try:
            return pd.to_datetime(value, format=DATE_FORMAT)
        except (ValueError, UnicodeError):
            self.fail(f"{value} is not a valid {DATE_FORMAT} date", param, ctx)

    def __repr__(self):
        return "TIMESTAMP"


def assert_valid_url(ctx, param, value):
    try:
        assert "https://" in value
    except ValueError:
        raise click.BadParameter(
            "URL to crawl must be on the format of https://XXXX.com"
        )
