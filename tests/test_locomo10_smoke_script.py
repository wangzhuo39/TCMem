import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "locomo10_smoke_test.py"
    spec = importlib.util.spec_from_file_location("locomo10_smoke_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_records_from_locomo_sample_pairs_turns_and_preserves_evidence_ids():
    smoke = _load_script_module()
    sample = {
        "sample_id": "conv-test",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "I joined a chess club."},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "That sounds fun."},
                {"speaker": "Alice", "dia_id": "D1:3", "text": "The first meeting is Friday."},
            ],
            "session_2_date_time": "7:55 pm on 9 June, 2023",
            "session_2": [
                {"speaker": "Bob", "dia_id": "D2:1", "text": "How was chess?"},
                {"speaker": "Alice", "dia_id": "D2:2", "text": "I liked the opening puzzles."},
            ],
        },
    }

    records = smoke.build_records_from_locomo_sample(sample, session_limit=2)
    assert [record.record_id for record in records] == [
        "rec_conv_test_s01_0001",
        "rec_conv_test_s01_0002",
        "rec_conv_test_s02_0001",
    ]
    assert records[0].session_uuid == "conv-test::session_1"
    assert records[0].current_time == "2023-05-08 13:56:00"
    assert records[0].source_turn_ids == ["D1:1", "D1:2"]
    assert records[1].assistant_content is None
    assert records[1].source_turn_ids == ["D1:3"]

    assert smoke.evidence_session_uuids(sample, ["D1:3", "D2:1"]) == [
        "conv-test::session_1",
        "conv-test::session_2",
    ]


def test_script_can_run_directly_from_repo_root():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "locomo10_smoke_test.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "locomo10.json" in result.stdout
