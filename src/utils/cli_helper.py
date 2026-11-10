from collections.abc import Callable
from typing import Any, cast, overload

import click
from click.core import Context


class SpecialHelpOrder(click.Group):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.help_priorities: dict[str | None, int] = {}
        super().__init__(*args, **kwargs)

    def list_commands(self, ctx: Context) -> list[str]:
        commands = super().list_commands(ctx)
        return sorted(commands, key=lambda command: (self.help_priorities.get(command, 99), command))

    @overload
    def command(self, __func: Callable[..., Any], /) -> click.Command: ...

    @overload
    def command(self, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], click.Command]: ...

    def command(self, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], click.Command] | click.Command:
        help_priority = kwargs.pop("help_priority", 99)
        if args and callable(args[0]):
            cmd = cast(click.Command, super().command(*args, **kwargs))
            self.help_priorities[cmd.name] = help_priority
            return cmd

        def decorator(f: Callable[..., Any]) -> click.Command:
            cmd = super(SpecialHelpOrder, self).command(*args, **kwargs)(f)
            self.help_priorities[cmd.name] = help_priority
            return cmd

        return decorator
