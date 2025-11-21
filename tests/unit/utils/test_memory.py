"""Unit tests for memory utilities."""

import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from arcaneum.utils.memory import (
    get_gpu_memory_info,
    estimate_safe_batch_size,
    get_available_memory_gb,
    calculate_safe_workers
)


class TestGetGpuMemoryInfo:
    """Tests for get_gpu_memory_info function."""

    def test_cuda_available(self):
        """Test CUDA GPU memory detection."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (8 * 1024**3, 16 * 1024**3)  # 8GB free, 16GB total
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict('sys.modules', {'torch': mock_torch}):
            available, total, device_type = get_gpu_memory_info()

        assert available == 8 * 1024**3
        assert total == 16 * 1024**3
        assert device_type == "cuda"

    def test_mps_available(self):
        """Test MPS (Apple Silicon) GPU memory detection."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.mps.driver_allocated_memory.return_value = 10 * 1024**3  # 10GB allocated

        # Mock psutil
        with patch('arcaneum.utils.memory.psutil') as mock_psutil:
            mock_memory = Mock()
            mock_memory.total = 32 * 1024**3  # 32GB RAM
            mock_psutil.virtual_memory.return_value = mock_memory

            with patch.dict('sys.modules', {'torch': mock_torch}):
                available, total, device_type = get_gpu_memory_info()

        expected_total = int(32 * 1024**3 * 0.7)  # 70% of RAM
        expected_available = expected_total - 10 * 1024**3

        assert available == expected_available
        assert total == expected_total
        assert device_type == "mps"

    def test_mps_fallback(self):
        """Test MPS fallback when driver_allocated_memory fails."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.mps.driver_allocated_memory.side_effect = Exception("Not available")

        # Mock psutil
        with patch('arcaneum.utils.memory.psutil') as mock_psutil:
            mock_memory = Mock()
            mock_memory.total = 32 * 1024**3  # 32GB RAM
            mock_psutil.virtual_memory.return_value = mock_memory

            with patch.dict('sys.modules', {'torch': mock_torch}):
                available, total, device_type = get_gpu_memory_info()

        expected_total = int(32 * 1024**3 * 0.7)  # 70% of RAM

        assert available == expected_total
        assert total == expected_total
        assert device_type == "mps"

    def test_no_gpu(self):
        """Test when no GPU is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict('sys.modules', {'torch': mock_torch}):
            available, total, device_type = get_gpu_memory_info()

        assert available is None
        assert total is None
        assert device_type is None

    def test_torch_not_installed(self):
        """Test when torch is not installed."""
        import sys
        # Save current torch module if it exists
        torch_backup = sys.modules.get('torch')

        # Remove torch from sys.modules to simulate not installed
        if 'torch' in sys.modules:
            del sys.modules['torch']

        # Mock the import to raise ImportError
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs:
                   (_ for _ in ()).throw(ImportError()) if name == 'torch' else __import__(name, *args, **kwargs)):
            available, total, device_type = get_gpu_memory_info()

        # Restore torch if it was there
        if torch_backup:
            sys.modules['torch'] = torch_backup

        assert available is None
        assert total is None
        assert device_type is None


class TestEstimateSafeBatchSize:
    """Tests for estimate_safe_batch_size function."""

    def test_1024d_model_with_8gb(self):
        """Test batch size estimation for 1024D model with 8GB free."""
        available_bytes = 8 * 1024**3  # 8GB
        model_dims = 1024

        batch_size = estimate_safe_batch_size(model_dims, available_bytes)

        # With safety factor 0.5: 4GB usable
        # 1024D needs 10MB per item: 4GB / 10MB = ~400
        assert 300 <= batch_size <= 500  # Reasonable range

    def test_1024d_model_with_limited_memory(self):
        """Test batch size estimation with limited memory."""
        available_bytes = 500 * 1024**2  # 500MB
        model_dims = 1024

        batch_size = estimate_safe_batch_size(model_dims, available_bytes)

        # With safety factor 0.5: 250MB usable
        # 1024D needs 10MB per item: 250MB / 10MB = 25
        assert 8 <= batch_size <= 50  # Should be floored at 8, but reasonably low

    def test_384d_model_scales_proportionally(self):
        """Test that smaller dimension models get proportionally larger batch sizes."""
        available_bytes = 4 * 1024**3  # 4GB
        model_dims_small = 384
        model_dims_large = 1024

        batch_size_small = estimate_safe_batch_size(model_dims_small, available_bytes)
        batch_size_large = estimate_safe_batch_size(model_dims_large, available_bytes)

        # Smaller model should allow larger batch size
        assert batch_size_small > batch_size_large

    def test_minimum_batch_size(self):
        """Test that batch size is floored at 8."""
        available_bytes = 10 * 1024**2  # 10MB (very small)
        model_dims = 1024

        batch_size = estimate_safe_batch_size(model_dims, available_bytes)

        assert batch_size >= 8

    def test_maximum_batch_size(self):
        """Test that batch size is capped at 1024."""
        available_bytes = 1000 * 1024**3  # 1TB (huge)
        model_dims = 384

        batch_size = estimate_safe_batch_size(model_dims, available_bytes)

        assert batch_size <= 1024

    def test_custom_safety_factor(self):
        """Test with custom safety factor."""
        available_bytes = 8 * 1024**3  # 8GB
        model_dims = 1024

        batch_size_conservative = estimate_safe_batch_size(model_dims, available_bytes, safety_factor=0.3)
        batch_size_aggressive = estimate_safe_batch_size(model_dims, available_bytes, safety_factor=0.8)

        # More conservative safety factor should give smaller batch size
        assert batch_size_conservative < batch_size_aggressive


class TestGetAvailableMemoryGb:
    """Tests for get_available_memory_gb function."""

    @patch('arcaneum.utils.memory.psutil')
    def test_returns_available_memory(self, mock_psutil):
        """Test that available memory is returned in GB."""
        mock_memory = Mock()
        mock_memory.available = 16 * 1024**3  # 16GB
        mock_psutil.virtual_memory.return_value = mock_memory

        result = get_available_memory_gb()

        assert result == pytest.approx(16.0, rel=0.01)


class TestEstimateSafeBatchSizeV2:
    """Tests for estimate_safe_batch_size_v2 function."""

    def test_stella_with_high_memory(self):
        """Test stella model with 10GB available memory."""
        from arcaneum.utils.memory import estimate_safe_batch_size_v2

        available_bytes = 10 * 1024**3  # 10GB
        batch_size = estimate_safe_batch_size_v2("stella", available_bytes)

        # stella: 2.5GB model + 2GB pipeline = 4.5GB fixed
        # Remaining: 5.5GB × 0.6 = 3.3GB usable
        # At 8MB per item: 3.3GB / 8MB ≈ 420
        assert 350 <= batch_size <= 500

    def test_jina_code_with_medium_memory(self):
        """Test jina-code model with 6GB available memory."""
        from arcaneum.utils.memory import estimate_safe_batch_size_v2

        available_bytes = 6 * 1024**3  # 6GB
        batch_size = estimate_safe_batch_size_v2("jina-code", available_bytes)

        # jina-code: 0.5GB model + 2GB pipeline = 2.5GB fixed
        # Remaining: 3.5GB × 0.6 = 2.1GB usable
        # At 5MB per item: 2.1GB / 5MB ≈ 430
        assert 350 <= batch_size <= 550

    def test_insufficient_memory(self):
        """Test when available memory is too low."""
        from arcaneum.utils.memory import estimate_safe_batch_size_v2

        available_bytes = 3 * 1024**3  # 3GB (less than stella model + pipeline)
        batch_size = estimate_safe_batch_size_v2("stella", available_bytes)

        # Should return minimum fallback
        assert batch_size == 8

    def test_unknown_model_uses_defaults(self):
        """Test that unknown model uses conservative defaults."""
        from arcaneum.utils.memory import estimate_safe_batch_size_v2

        available_bytes = 10 * 1024**3  # 10GB
        batch_size = estimate_safe_batch_size_v2("unknown-model", available_bytes)

        # Should use default model weights (2.0GB) and activation (8MB)
        # 10GB - 2GB model - 2GB pipeline = 6GB × 0.6 = 3.6GB / 8MB ≈ 460
        assert 350 <= batch_size <= 600

    def test_custom_pipeline_overhead(self):
        """Test with custom pipeline overhead."""
        from arcaneum.utils.memory import estimate_safe_batch_size_v2

        available_bytes = 10 * 1024**3  # 10GB
        batch_size = estimate_safe_batch_size_v2(
            "jina-code",
            available_bytes,
            pipeline_overhead_gb=1.0  # Less overhead
        )

        # More usable memory should give larger batch size
        assert batch_size > 400

    def test_custom_safety_factor(self):
        """Test with custom safety factor."""
        from arcaneum.utils.memory import estimate_safe_batch_size_v2

        available_bytes = 10 * 1024**3  # 10GB

        batch_conservative = estimate_safe_batch_size_v2(
            "stella", available_bytes, safety_factor=0.4
        )
        batch_aggressive = estimate_safe_batch_size_v2(
            "stella", available_bytes, safety_factor=0.8
        )

        # More aggressive safety factor should give larger batch
        assert batch_aggressive > batch_conservative

    def test_clamped_to_maximum(self):
        """Test that batch size is capped at 1024."""
        from arcaneum.utils.memory import estimate_safe_batch_size_v2

        available_bytes = 100 * 1024**3  # 100GB (unrealistic but tests cap)
        batch_size = estimate_safe_batch_size_v2("bge-small", available_bytes)

        # Should be capped at maximum
        assert batch_size == 1024

    def test_bge_small_uses_less_memory(self):
        """Test that smaller models allow larger batches."""
        from arcaneum.utils.memory import estimate_safe_batch_size_v2

        available_bytes = 8 * 1024**3  # 8GB

        batch_stella = estimate_safe_batch_size_v2("stella", available_bytes)
        batch_small = estimate_safe_batch_size_v2("bge-small", available_bytes)

        # bge-small has smaller model (0.3GB) and lower activation (3MB)
        # Should allow larger batch size
        assert batch_small > batch_stella


class TestCalculateSafeWorkers:
    """Tests for calculate_safe_workers function."""

    @patch('arcaneum.utils.memory.psutil')
    def test_sufficient_memory(self, mock_psutil):
        """Test when there's sufficient memory for requested workers."""
        mock_memory = Mock()
        mock_memory.available = 16 * 1024**3  # 16GB
        mock_memory.total = 32 * 1024**3  # 32GB
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory

        workers, warning = calculate_safe_workers(
            requested_workers=8,
            estimated_memory_per_worker_mb=1024  # 1GB per worker
        )

        # Should be able to fit 8 workers (8GB needed, 16GB available * 0.8 = 12.8GB usable)
        assert workers == 8
        assert warning == ""

    @patch('arcaneum.utils.memory.psutil')
    def test_insufficient_memory(self, mock_psutil):
        """Test when there's insufficient memory for requested workers."""
        mock_memory = Mock()
        mock_memory.available = 4 * 1024**3  # 4GB
        mock_memory.total = 8 * 1024**3  # 8GB
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory

        workers, warning = calculate_safe_workers(
            requested_workers=8,
            estimated_memory_per_worker_mb=1024  # 1GB per worker
        )

        # Can only fit 3 workers (4GB * 0.8 = 3.2GB usable / 1GB per worker = 3)
        assert workers == 3
        assert "Reduced workers from 8 to 3" in warning
        assert "memory constraints" in warning
