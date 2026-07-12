import os
import click
from click.core import Command, Context


class CLI(click.MultiCommand):

    def list_commands(self, ctx: Context):

        plugins = []
        for dir_name in os.listdir(__package__):
            dir_path = os.path.join(__package__, dir_name)
            if os.path.isdir(dir_path):
                for filename in os.listdir(dir_path):
                    if filename == "cli.py":
                        plugins.append(os.path.basename(dir_path))

        plugins.sort()
        return plugins

    def get_command(self, ctx: Context, cmd_name: str) -> Command | None:

        commands = {}
        filename = os.path.join(__package__, cmd_name, "cli.py")
        try:
            with open(filename) as file:
                code = compile(file.read(), filename, "exec")
                eval(code, commands, commands)
            return commands["cli"]
        except FileNotFoundError as e:
            print(f"Command {e} not found")


@click.command(
    cls=CLI,
    context_settings=dict(help_option_names=["-h", "--help"], max_content_width=120),
)
@click.version_option(version="0.1.12")
def main():
    "WELCOME TO PACKAGE HART"
    pass


if __name__ == "__main__":
    main()
