"""Tests for formatting utilities."""

import pytest
from arcaneum.utils.formatting import format_duration, format_size


class TestFormatSize:
    """Tests for format_size function."""

    def test_bytes(self):
        """Test formatting values under 1 KB."""
        assert format_size(0) == "0.0 B"
        assert format_size(1) == "1.0 B"
        assert format_size(512) == "512.0 B"
        assert format_size(1023) == "1023.0 B"

    def test_kilobytes(self):
        """Test formatting values in KB range."""
        assert format_size(1024) == "1.0 KB"
        assert format_size(1536) == "1.5 KB"
        assert format_size(10240) == "10.0 KB"
        assert format_size(1048575) == "1024.0 KB"

    def test_megabytes(self):
        """Test formatting values in MB range."""
        assert format_size(1048576) == "1.0 MB"
        assert format_size(1572864) == "1.5 MB"
        assert format_size(10485760) == "10.0 MB"
        assert format_size(104857600) == "100.0 MB"

    def test_gigabytes(self):
        """Test formatting values in GB range."""
        assert format_size(1073741824) == "1.0 GB"
        assert format_size(1610612736) == "1.5 GB"
        assert format_size(10737418240) == "10.0 GB"

    def test_terabytes(self):
        """Test formatting values in TB range."""
        assert format_size(1099511627776) == "1.0 TB"
        assert format_size(1649267441664) == "1.5 TB"

    def test_petabytes(self):
        """Test formatting values in PB range (edge case)."""
        # 1 PB = 1024 TB
        assert format_size(1125899906842624) == "1.0 PB"

    def test_float_input(self):
        """Test that float inputs are handled correctly."""
        assert format_size(1024.0) == "1.0 KB"
        assert format_size(1536.5) == "1.5 KB"


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
