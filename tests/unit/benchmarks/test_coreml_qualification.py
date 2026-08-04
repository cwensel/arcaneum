from pathlib import Path

import numpy as np

from arcaneum.benchmarks.coreml import coreml_provider_options, route_bucketed


def test_coreml_options_request_cached_static_mlprogram(tmp_path: Path):
    options = coreml_provider_options(tmp_path)
    assert options["ModelFormat"] == "MLProgram"
    assert options["RequireStaticInputShapes"] == "1"
    assert options["SpecializationStrategy"] == "FastPrediction"
    assert options["ProfileComputePlan"] == "1"
    assert options["ModelCacheDirectory"] == str(tmp_path.resolve())


def test_bucket_routing_pads_and_restores_output_order():
    calls = []

    def encode(texts):
        calls.append(list(texts))
        return np.asarray([[len(text), ord(text[0])] for text in texts], dtype=np.float32)

    actual, buckets = route_bucketed(["a", "longest", "mid"], encode, buckets=(1, 2, 4))
    assert buckets == [4]
    assert len(calls[0]) == 4 and calls[0][-1] == calls[0][-2]
    assert actual[:, 0].tolist() == [1, 7, 3]
    assert actual[:, 1].tolist() == [ord("a"), ord("l"), ord("m")]


def test_bucket_routing_splits_at_largest_bucket():
    actual, buckets = route_bucketed(
        [str(i) for i in range(5)],
        lambda texts: np.asarray([[int(text)] for text in texts], dtype=np.float32),
        buckets=(1, 2, 4),
    )
    assert buckets == [4, 1]
    assert actual[:, 0].tolist() == [0, 1, 2, 3, 4]
