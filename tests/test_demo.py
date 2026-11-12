import json
from pathlib import Path

import pytest

from turnaware_recsys.demo import run_demo


@pytest.mark.parametrize("with_dense", [False, True])
def test_end_to_end_oof_and_turn_routing(tmp_path, with_dense):
    pytest.importorskip("gensim")
    data_path = Path(__file__).parents[1] / "examples" / "synthetic_data.json"
    data = json.loads(data_path.read_text())
    embeddings_path = None
    if with_dense:
        groups = {"r": [1, 0, 0], "j": [0, 1, 0], "e": [0, 0, 1]}
        vectors = {
            "tracks": {r["track_id"]: groups[r["track_id"][0]] for r in data["catalog"]},
            "queries": [
                {
                    "session_id": q["session_id"],
                    "turn_number": q["turn_number"],
                    "vector": groups[q["target"][0]],
                }
                for q in data["train_queries"] + data["evaluation_queries"]
            ],
        }
        # These vectors are fictional integration-test inputs, never competition embeddings.
        embeddings_path = tmp_path / "vectors.json"
        embeddings_path.write_text(json.dumps(vectors))
    result = run_demo(data_path, embeddings_path=embeddings_path)
    assert result["candidate_sources"] == ["item2vec", "artist_history"]
    assert result["oof_training_queries"] == 6
    assert result["metrics"]["turn_1"]["count"] == 1
    assert ("dense" in result["ranking_sources"]) == with_dense
