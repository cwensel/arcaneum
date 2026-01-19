"""Tests for command_wrapper module."""

import sys
import pytest
from unittest.mock import patch, MagicMock

from arcaneum.cli.core.command_wrapper import command_context, with_error_handling
from arcaneum.cli.errors import InvalidArgumentError, ResourceNotFoundError


class TestCommandContext:
    """Tests for command_context context manager."""

    @patch('arcaneum.cli.core.command_wrapper.interaction_logger')
    def test_successful_execution(self, mock_logger):
        """Test that successful execution logs start and finish."""
        with command_context("test", "operation"):
            pass

        mock_logger.start.assert_called_once_with("test", "operation")
        mock_logger.finish.assert_called_once_with()

    @patch('arcaneum.cli.core.command_wrapper.interaction_logger')
    def test_logs_extra_kwargs(self, mock_logger):
        """Test that extra kwargs are passed to logger.start."""
        with command_context("collection", "create", collection="MyCollection"):
            pass

        mock_logger.start.assert_called_once_with(
            "collection", "create", collection="MyCollection"
        )

    @patch('arcaneum.cli.core.command_wrapper.interaction_logger')
    def test_invalid_argument_error_reraises(self, mock_logger):
        """Test that InvalidArgumentError is re-raised after logging."""
        with pytest.raises(InvalidArgumentError):
            with command_context("test", "operation"):
                raise InvalidArgumentError("Invalid argument")

        mock_logger.finish.assert_called_once()
        # Check error was logged
        call_kwargs = mock_logger.finish.call_args[1]
        assert 'error' in call_kwargs

    @patch('arcaneum.cli.core.command_wrapper.interaction_logger')
    def test_resource_not_found_error_reraises(self, mock_logger):
        """Test that ResourceNotFoundError is re-raised after logging."""
        with pytest.raises(ResourceNotFoundError):
            with command_context("test", "operation"):
                raise ResourceNotFoundError("Resource not found")

        mock_logger.finish.assert_called_once()
        call_kwargs = mock_logger.finish.call_args[1]
        assert 'error' in call_kwargs

    @patch('arcaneum.cli.core.command_wrapper.print_error')
    @patch('arcaneum.cli.core.command_wrapper.interaction_logger')
    def test_generic_exception_exits(self, mock_logger, mock_print_error):
        """Test that generic exception prints error and exits."""
        with pytest.raises(SystemExit) as exc_info:
            with command_context("test", "operation", output_json=False):
                raise RuntimeError("Something went wrong")

        assert exc_info.value.code == 1
        mock_print_error.assert_called_once()
        error_msg = mock_print_error.call_args[0][0]
        assert "Something went wrong" in error_msg

    @patch('arcaneum.cli.core.command_wrapper.print_error')
    @patch('arcaneum.cli.core.command_wrapper.interaction_logger')
    def test_custom_error_prefix(self, mock_logger, mock_print_error):
        """Test that custom error prefix is used."""
        with pytest.raises(SystemExit):
            with command_context("test", "operation", error_prefix="Failed to create"):
                raise RuntimeError("Error details")

        error_msg = mock_print_error.call_args[0][0]
        assert "Failed to create" in error_msg


class TestWithErrorHandling:
    """Tests for with_error_handling decorator."""

    @patch('arcaneum.cli.core.command_wrapper.interaction_logger')
    def test_decorated_function_executes(self, mock_logger):
        """Test that decorated function executes normally."""
        @with_error_handling("test", "operation")
        def my_command(name: str, output_json: bool):
            return f"Result: {name}"

        result = my_command("test-name", output_json=False)
        assert result == "Result: test-name"

    @patch('arcaneum.cli.core.command_wrapper.interaction_logger')
    def test_logs_operation(self, mock_logger):
        """Test that decorator logs namespace and operation."""
        @with_error_handling("collection", "create")
        def create_command(name: str, output_json: bool):
            pass

        create_command("MyCollection", output_json=False)

        mock_logger.start.assert_called_once()
        call_args = mock_logger.start.call_args[0]
        assert call_args[0] == "collection"
        assert call_args[1] == "create"

    @patch('arcaneum.cli.core.command_wrapper.interaction_logger')
    def test_logs_specified_params(self, mock_logger):
        """Test that decorator logs specified parameter values."""
        @with_error_handling("collection", "create", log_params=["name"])
        def create_command(name: str, model: str, output_json: bool):
            pass

        create_command("MyCollection", "stella", output_json=False)

        mock_logger.start.assert_called_once()
        call_kwargs = mock_logger.start.call_args[1]
        assert call_kwargs.get("name") == "MyCollection"

    @patch('arcaneum.cli.core.command_wrapper.interaction_logger')
    def test_logs_kwargs_params(self, mock_logger):
        """Test that decorator logs kwargs parameter values."""
        @with_error_handling("collection", "create", log_params=["name"])
        def create_command(name: str, output_json: bool):
            pass

        create_command(name="MyCollection", output_json=False)

        call_kwargs = mock_logger.start.call_args[1]
        assert call_kwargs.get("name") == "MyCollection"

    @patch('arcaneum.cli.core.command_wrapper.interaction_logger')
    def test_invalid_argument_error_reraises(self, mock_logger):
        """Test that InvalidArgumentError is re-raised from decorated function."""
        @with_error_handling("test", "operation")
        def failing_command(output_json: bool):
            raise InvalidArgumentError("Bad argument")

        with pytest.raises(InvalidArgumentError):
            failing_command(output_json=False)

    @patch('arcaneum.cli.core.command_wrapper.print_error')
    @patch('arcaneum.cli.core.command_wrapper.interaction_logger')
    def test_generic_error_exits(self, mock_logger, mock_print_error):
        """Test that generic exception causes sys.exit(1)."""
        @with_error_handling("test", "operation")
        def failing_command(output_json: bool):
            raise RuntimeError("Unexpected error")

        with pytest.raises(SystemExit) as exc_info:
            failing_command(output_json=False)

        assert exc_info.value.code == 1

    @patch('arcaneum.cli.core.command_wrapper.interaction_logger')
    def test_preserves_function_metadata(self, mock_logger):
        """Test that decorator preserves function name and docstring."""
        @with_error_handling("test", "operation")
        def documented_command(output_json: bool):
            """This is the docstring."""
            pass

        assert documented_command.__name__ == "documented_command"
        assert "docstring" in documented_command.__doc__
