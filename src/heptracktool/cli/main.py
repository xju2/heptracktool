import click
import importlib
import pkgutil
import inspect
import heptracktool.cli


@click.group()
def main():
    """heptracktool: Tools for high energy physics particle tracking."""


# --- Auto-discover subcommands ---
def _register_commands():
    package = heptracktool.cli

    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        if module_name == "main":
            continue  # skip main.py itself
        module = importlib.import_module(f"{package.__name__}.{module_name}")

        # find all click commands in the module
        for obj_name, obj in inspect.getmembers(module):
            if isinstance(obj, click.core.Command):
                main.add_command(obj)


# Run discovery
_register_commands()
