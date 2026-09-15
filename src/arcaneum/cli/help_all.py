"""Render a complete, live description of the Arcaneum CLI."""

from __future__ import annotations

from typing import Any

import click


def _child_context(parent: click.Context, command: click.Command, name: str) -> click.Context:
    return click.Context(
        command,
        parent=parent,
        info_name=name,
        terminal_width=parent.terminal_width,
        max_content_width=parent.max_content_width,
        color=parent.color,
    )


def _visible_children(
    command: click.Command, ctx: click.Context
) -> list[tuple[str, click.Command]]:
    if not isinstance(command, click.Group):
        return []

    children = []
    for name in command.list_commands(ctx):
        child = command.get_command(ctx, name)
        if child is not None and not child.hidden:
            children.append((name, child))
    return children


def render_help_all(ctx: click.Context) -> str:
    """Return standard Click help for every visible command in the tree."""
    sections: list[str] = []

    def visit(command: click.Command, command_ctx: click.Context) -> None:
        help_text = command.get_help(command_ctx).rstrip()
        sections.append(f"=== {command_ctx.command_path} ===\n{help_text}")

        for name, child in _visible_children(command, command_ctx):
            child_ctx = _child_context(command_ctx, child, name)
            with child_ctx.scope(cleanup=False):
                visit(child, child_ctx)

    visit(ctx.command, ctx)
    return "\n\n".join(sections) + "\n"


def build_help_manifest(ctx: click.Context, *, version: str) -> dict[str, Any]:
    """Return a compact, machine-readable description of the command tree."""
    commands: list[dict[str, Any]] = []

    def describe_param(param: click.Parameter) -> dict[str, Any]:
        click_info = param.to_info_dict()
        description: dict[str, Any] = {
            "name": param.name,
            "type": click_info["type"],
            "required": param.required,
        }

        if isinstance(param, click.Option):
            description["flags"] = [*param.opts, *param.secondary_opts]
            if param.help:
                description["help"] = param.help
            if param.is_flag:
                description["is_flag"] = True
            if param.count:
                description["count"] = True

        if param.multiple:
            description["multiple"] = True
        if param.nargs != 1:
            description["nargs"] = param.nargs
        if click_info["default"] is not None:
            description["default"] = click_info["default"]
        if param.envvar:
            description["envvar"] = param.envvar
        if getattr(param, "deprecated", False):
            description["deprecated"] = param.deprecated

        return description

    def visit(command: click.Command, command_ctx: click.Context) -> None:
        children = _visible_children(command, command_ctx)
        params = [
            param
            for param in command.get_params(command_ctx)
            if not getattr(param, "hidden", False)
        ]
        description: dict[str, Any] = {
            "path": command_ctx.command_path,
            "usage": command.get_usage(command_ctx).removeprefix("Usage: ").strip(),
            "options": [
                describe_param(param) for param in params if isinstance(param, click.Option)
            ],
            "arguments": [
                describe_param(param) for param in params if isinstance(param, click.Argument)
            ],
        }
        if command.help:
            description["description"] = command.help
        if children:
            description["subcommands"] = [name for name, _ in children]
        if command.deprecated:
            description["deprecated"] = command.deprecated
        commands.append(description)

        for name, child in children:
            child_ctx = _child_context(command_ctx, child, name)
            with child_ctx.scope(cleanup=False):
                visit(child, child_ctx)

    visit(ctx.command, ctx)
    return {
        "schema_version": 1,
        "program": ctx.info_name or ctx.command.name,
        "version": version,
        "commands": commands,
    }
