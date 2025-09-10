from __future__ import annotations
import os
import shlex
from typing import Union, Tuple, Set


class CommandValidator:
    """Simple validator to ensure basic command security"""

    # Most dangerous shell operations that shouldn't appear in legitimate commands
    DANGEROUS_OPERATIONS = [
        "&&",  # Command chaining
        "||",  # Command chaining
        " ; ",  # Command chaining (with spaces to avoid ffmpeg filter syntax)
        ";\\n",  # Command chaining (newline variant)
        "$(",
        "`",  # Command substitution
        "> /",  # Writing to root
        ">>/",  # Appending to root
        "sudo ",  # Privilege escalation (with space to avoid false positives)
        "| rm",  # Pipe to remove
        "| sh",  # Pipe to shell
        "| bash",  # Pipe to shell
        "eval ",  # Command evaluation (with space)
        "exec ",  # Command execution (with space)
    ]

    def __init__(self, allowed_commands: Set[str]):
        """
        Initialize the command validator.

        Args:
            allowed_commands: Set of base commands that are allowed to be executed
        """
        self.allowed_commands = {cmd.lower() for cmd in allowed_commands}

    def validate_command(self, command: str) -> Tuple[bool, Union[str, None]]:
        """
        Validates that a command is safe to execute.
        Only checks for base command and the most dangerous shell operations.

        Args:
            command: The command string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not command or not isinstance(command, str):
            return False, "Command must be a non-empty string"

        # Try to parse command into tokens and get base command
        try:
            tokens = shlex.split(command)
            if not tokens:
                return False, "Empty command"
            base_cmd = os.path.basename(tokens[0]).lower()
        except ValueError as e:
            return False, f"Invalid command syntax: {str(e)}"

        # Verify base command is allowed
        if base_cmd not in self.allowed_commands:
            return False, f"Command '{base_cmd}' is not in the allowed list"

        # Check for dangerous operations
        for pattern in self.DANGEROUS_OPERATIONS:
            if pattern in command:
                return False, f"Command contains dangerous operation: {pattern}"

        return True, None


def get_human_readable_error(error_list):
    errors = [f"{error['loc'][0]}: {error['msg']}" for error in error_list]
    error_str = "\n\t".join(errors)
    error_str = f"Invalid args\n\t{error_str}"
    return error_str


def is_downloadable_file(value):
    import replicate
    return isinstance(value, replicate.helpers.FileOutput) or (
        isinstance(value, str)
        and (
            os.path.isfile(value)  # is a file
            or (  # is a url but not from twitter
                value.startswith(("http://", "https://"))
                and "x.com" not in value
                and "pbs.twimg.com" not in value
                and not any(value.endswith(ext) for ext in [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", "html", "htm"])
            )
        )
    )