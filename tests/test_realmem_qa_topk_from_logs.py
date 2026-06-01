import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from tcmem.evals.realmem_qa_topk_from_logs import build_qa_topk_result, evaluate_qa_topk_from_logs, parse_args
from tcmem.evals.realmem_top_session import QueryExample
from tcmem.prompts import PromptRegistry


class RealMemQATopKFromLogsTest(unittest.TestCase):
    def test_build_qa_topk_result_generates_and_scores_from_top_k_sessions(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def generate(self, prompt: str, **_kwargs) -> str:
                self.prompts.append(prompt)
                if prompt.startswith("ANSWER"):
                    return "answer from top sessions"
                return json.dumps({"score": 2, "reason": "uses the first sessions"})

        registry = PromptRegistry.from_mapping(
            {
                "realmem_answer_generation": {
                    "system": "",
                    "user": "ANSWER {{evidence_text}}",
                },
                "realmem_qa_judge": {
                    "system": "",
                    "user": "JUDGE {{candidate_answer}} AGAINST {{gold_memory_text}}",
                },
            }
        )
        logged = {
            "question": "question",
            "ranked_sessions": [{"session_uuid": "s1"}, {"session_uuid": "s2"}, {"session_uuid": "s3"}],
        }
        example = QueryExample(
            query_id="Q-0001",
            session_identifier="session",
            session_uuid="query-session",
            current_time="2026-05-31",
            turn_index=0,
            question="question",
            reference_answer="reference",
            gold_session_uuids=["s1"],
            gold_memory_text="gold memory",
            memory_used=[],
        )

        result = build_qa_topk_result(
            qid="Q-0001",
            logged=logged,
            example=example,
            session_text_by_uuid={"s1": "User: first", "s2": "User: second", "s3": "User: third"},
            top_k=2,
            client=FakeClient(),
            qa_model_name="judge-model",
            prompt_registry=registry,
        )

        self.assertEqual(result["qa_score"], 2)
        self.assertEqual(result["generated_answer"], "answer from top sessions")
        self.assertEqual(result["evidence_session_uuids"], ["s1", "s2"])
        self.assertIn("session_uuid=s1", result["evidence_used"])
        self.assertIn("session_uuid=s2", result["evidence_used"])
        self.assertNotIn("session_uuid=s3", result["evidence_used"])
        self.assertEqual(result["source_qa_score"], None)

    def test_evaluate_qa_topk_from_logs_writes_supplemental_output(self) -> None:
        class FakeClient:
            def __init__(self, **_kwargs) -> None:
                pass

            def generate(self, prompt: str, **_kwargs) -> str:
                if prompt.startswith("ANSWER"):
                    return "generated"
                return json.dumps({"score": 3, "reason": "complete"})

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            dataset_path = base / "dataset.json"
            run_dir = base / "run"
            run_dir.mkdir()
            dataset_path.write_text(
                json.dumps(
                    {
                        "dialogues": [
                            {
                                "session_identifier": "session-1",
                                "session_uuid": "s-query",
                                "current_time": "2026-05-31",
                                "dialogue_turns": [
                                    {"speaker": "User", "content": "question", "is_query": True, "query_id": "Q-0001"},
                                    {
                                        "speaker": "Assistant",
                                        "content": "reference",
                                        "memory_session_uuids": ["s1"],
                                        "memory_used": [{"session_uuid": "s1", "content": "gold memory"}],
                                    },
                                ],
                            },
                            {
                                "session_identifier": "session-2",
                                "session_uuid": "s1",
                                "current_time": "2026-05-30",
                                "dialogue_turns": [
                                    {"speaker": "User", "content": "first"},
                                    {"speaker": "Assistant", "content": "first answer"},
                                ],
                            },
                            {
                                "session_identifier": "session-3",
                                "session_uuid": "s2",
                                "current_time": "2026-05-30",
                                "dialogue_turns": [
                                    {"speaker": "User", "content": "second"},
                                    {"speaker": "Assistant", "content": "second answer"},
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            query_payload = {
                "query_id": "Q-0001",
                "question": "question",
                "ranked_sessions": [{"session_uuid": "s1"}, {"session_uuid": "s2"}],
                "qa_score": 1,
            }
            (run_dir / "query_results.jsonl").write_text(
                json.dumps({"payload": query_payload}) + "\n",
                encoding="utf-8",
            )
            out_file = base / "qa_top1.json"
            prompt_dir = base / "prompts"
            prompt_dir.mkdir()
            (prompt_dir / "default_prompts.yaml").write_text(
                """
realmem_answer_generation:
  system: ""
  user: "ANSWER {{evidence_text}}"
realmem_qa_judge:
  system: ""
  user: "JUDGE {{candidate_answer}}"
""".strip(),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--dataset",
                    str(dataset_path),
                    "--api-key",
                    "test-key",
                    "--top-k",
                    "1",
                    "--out-file",
                    str(out_file),
                    "--prompt-path",
                    str(prompt_dir),
                    "--max-workers",
                    "1",
                    "--verbose",
                ]
            )

            stdout = StringIO()
            with patch("tcmem.evals.realmem_qa_topk_from_logs.OpenAICompatibleLLMClient", FakeClient):
                with redirect_stdout(stdout):
                    result = evaluate_qa_topk_from_logs(args)

            output = json.loads(out_file.read_text(encoding="utf-8"))

        self.assertEqual(result["summary"]["average_qa_score"], 3.0)
        self.assertEqual(output["detailed_results"][0]["evidence_session_uuids"], ["s1"])
        self.assertEqual(output["detailed_results"][0]["source_qa_score"], 1)
        self.assertIn("[qa-top1] selected=1", stdout.getvalue())
        self.assertIn("[qa-top1] start 1/1 Q-0001", stdout.getvalue())
        self.assertIn("[qa-top1] done 1/1 Q-0001 score=3", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
