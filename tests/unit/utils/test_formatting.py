"""Tests for formatting utilities."""

import pytest
from arcaneum.utils.formatting import format_duration


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_seconds_only(self):
        """Test formatting values under 60 seconds."""
        assert format_duration(0) == "0s"
        assert format_duration(1) == "1s"
        assert format_duration(45) == "45s"
        assert format_duration(59) == "59s"
        assert format_duration(59.9) == "59s"  # Truncates, doesn't round

    def test_minutes_and_seconds(self):
        """Test formatting values between 1 minute and 1 hour."""
        assert format_duration(60) == "1m"
        assert format_duration(61) == "1m 1s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(185) == "3m 5s"
        assert format_duration(600) == "10m"
        assert format_duration(3599) == "59m 59s"

    def test_hours_and_minutes(self):
        """Test formatting values of 1 hour or more."""
        assert format_duration(3600) == "1h"
        assert format_duration(3660) == "1h 1m"
        assert format_duration(3725) == "1h 2m"
        assert format_duration(7200) == "2h"
        assert format_duration(7320) == "2h 2m"
        assert format_duration(36000) == "10h"
        assert format_duration(36061) == "10h 1m"

    def test_float_input(self):
        """Test that float inputs are handled correctly (truncated to int)."""
        assert format_duration(45.7) == "45s"
        assert format_duration(90.9) == "1m 30s"
        assert format_duration(3661.5) == "1h 1m"
