"""Unit tests for arcaneum.monitoring.pipeline_profiler."""

import time
import threading
import pytest

from arcaneum.monitoring.pipeline_profiler import (
    PipelineProfiler,
    StageMetrics,
    create_profiler,
)


# --- StageMetrics ---

class TestStageMetrics:
    def test_duration(self):
        m = StageMetrics(name="test", start_time=1.0, end_time=3.5)
        assert m.duration == pytest.approx(2.5)

    def test_throughput(self):
        m = StageMetrics(name="test", start_time=0.0, end_time=2.0, items_processed=100)
        assert m.throughput == pytest.approx(50.0)

    def test_throughput_zero_duration(self):
        m = StageMetrics(name="test", start_time=1.0, end_time=1.0, items_processed=50)
        assert m.throughput == 0.0


# --- PipelineProfiler.stage() context manager ---

class TestStageContextManager:
    def test_records_duration(self):
        profiler = PipelineProfiler()
        with profiler.stage("embedding"):
            time.sleep(0.01)
        assert "embedding" in profiler.stages
        assert profiler.stages["embedding"].duration >= 0.01

    def test_records_item_count(self):
        profiler = PipelineProfiler()
        with profiler.stage("embedding", item_count=500):
            pass
        assert profiler.stages["embedding"].items_processed == 500

    def test_stage_overwrites_previous(self):
        profiler = PipelineProfiler()
        with profiler.stage("s1"):
            pass
        with profiler.stage("s1"):
            pass
        assert len(profiler.stages) == 1

    def test_multiple_stages(self):
        profiler = PipelineProfiler()
        with profiler.stage("a"):
            pass
        with profiler.stage("b"):
            pass
        assert "a" in profiler.stages
        assert "b" in profiler.stages

    def test_stage_records_on_exception(self):
        profiler = PipelineProfiler()
        with pytest.raises(ValueError):
            with profiler.stage("failing"):
                raise ValueError("boom")
        assert "failing" in profiler.stages


# --- record_stage ---

class TestRecordStage:
    def test_records_single_call(self):
        profiler = PipelineProfiler()
        profiler.record_stage("upload", duration=2.5, item_count=100)
        assert profiler.stages["upload"].duration == pytest.approx(2.5)
        assert profiler.stages["upload"].items_processed == 100

    def test_accumulates_multiple_calls(self):
        profiler = PipelineProfiler()
        profiler.record_stage("upload", duration=1.0, item_count=50)
        profiler.record_stage("upload", duration=2.0, item_count=75)
        assert profiler.stages["upload"].duration == pytest.approx(3.0)
        assert profiler.stages["upload"].items_processed == 125


# --- total_duration and elapsed_time ---

class TestDuration:
    def test_total_duration_sums_stages(self):
        profiler = PipelineProfiler()
        profiler.record_stage("a", duration=1.0)
        profiler.record_stage("b", duration=2.0)
        assert profiler.total_duration == pytest.approx(3.0)

    def test_total_duration_empty(self):
        profiler = PipelineProfiler()
        assert profiler.total_duration == 0.0

    def test_elapsed_time_without_start(self):
        profiler = PipelineProfiler()
        assert profiler.elapsed_time == 0.0

    def test_elapsed_time_with_start_stop(self):
        profiler = PipelineProfiler()
        profiler.start()
        time.sleep(0.01)
        profiler.stop()
        assert profiler.elapsed_time >= 0.01


# --- report ---

class TestReport:
    def test_report_contains_header(self):
        profiler = PipelineProfiler()
        profiler.record_stage("embed", duration=1.0)
        report = profiler.report()
        assert "Pipeline Performance Report" in report
        assert "Total:" in report

    def test_report_no_header(self):
        profiler = PipelineProfiler()
        profiler.record_stage("embed", duration=1.0)
        report = profiler.report(include_header=False)
        assert "Pipeline Performance Report" not in report
        assert "Total:" in report

    def test_report_contains_stage_name(self):
        profiler = PipelineProfiler()
        profiler.record_stage("my_stage", duration=2.0, item_count=100)
        report = profiler.report()
        assert "my_stage" in report
        assert "items/s" in report

    def test_report_no_item_count_omits_throughput(self):
        profiler = PipelineProfiler()
        profiler.record_stage("my_stage", duration=2.0)
        report = profiler.report()
        assert "items/s" not in report

    def test_report_shows_percentage(self):
        profiler = PipelineProfiler()
        profiler.record_stage("a", duration=1.0)
        profiler.record_stage("b", duration=3.0)
        report = profiler.report()
        assert "25.0%" in report
        assert "75.0%" in report


# --- get_stage_summary ---

class TestGetStageSummary:
    def test_known_stage(self):
        profiler = PipelineProfiler()
        profiler.record_stage("embed", duration=2.0, item_count=100)
        summary = profiler.get_stage_summary("embed")
        assert summary is not None
        assert "embed" in summary

    def test_unknown_stage_returns_none(self):
        profiler = PipelineProfiler()
        assert profiler.get_stage_summary("missing") is None


# --- get_compact_summary ---

class TestGetCompactSummary:
    def test_returns_string(self):
        profiler = PipelineProfiler()
        profiler.record_stage("embedding", duration=2.0)
        summary = profiler.get_compact_summary()
        assert isinstance(summary, str)
        assert "Profile:" in summary

    def test_preferred_stage_names(self):
        profiler = PipelineProfiler()
        profiler.record_stage("file_processing", duration=1.0)
        profiler.record_stage("embedding", duration=2.0)
        profiler.record_stage("upload", duration=0.5)
        summary = profiler.get_compact_summary()
        assert "file_" in summary
        assert "embed" in summary
        assert "uploa" in summary

    def test_fallback_for_other_stages(self):
        profiler = PipelineProfiler()
        profiler.record_stage("custom_stage", duration=1.0)
        summary = profiler.get_compact_summary()
        assert "custom_s" in summary


# --- reset ---

class TestReset:
    def test_clears_stages(self):
        profiler = PipelineProfiler()
        profiler.record_stage("a", duration=1.0)
        profiler.reset()
        assert profiler.stages == {}

    def test_clears_timing(self):
        profiler = PipelineProfiler()
        profiler.start()
        profiler.stop()
        profiler.reset()
        assert profiler.elapsed_time == 0.0

    def test_usable_after_reset(self):
        profiler = PipelineProfiler()
        profiler.record_stage("a", duration=1.0)
        profiler.reset()
        profiler.record_stage("b", duration=2.0)
        assert "b" in profiler.stages
        assert "a" not in profiler.stages


# --- Thread safety ---

class TestThreadSafety:
    def test_concurrent_record_stage(self):
        profiler = PipelineProfiler()
        errors = []

        def record():
            try:
                for _ in range(50):
                    profiler.record_stage("shared", duration=0.001, item_count=1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert profiler.stages["shared"].items_processed == 500

    def test_concurrent_stage_context_manager(self):
        profiler = PipelineProfiler()
        errors = []

        def run_stage(name):
            try:
                with profiler.stage(name):
                    pass
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_stage, args=(f"stage_{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(profiler.stages) == 20


# --- create_profiler factory ---

def test_create_profiler_returns_instance():
    profiler = create_profiler()
    assert isinstance(profiler, PipelineProfiler)
    assert profiler.stages == {}
