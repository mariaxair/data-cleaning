"""Offline behavioral checks for the paid APS workflow."""

import json
import uuid
from pathlib import Path

import pytest

import aps_generate_qa as qa


@pytest.fixture
def article():
    return {"uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, "aps-test")),
            "text": "Original article.\r\n\r\nDetails in the original source.\r\n",
            "title": "Article", "lang": "en", "source_url": "https://example.invalid/article",
            "article_id": "test", "published": "2026-01-01", "category": "Economy"}


@pytest.fixture
def templates():
    return {role: qa.Template((Path(qa.__file__).parent / "prompts" / f"aps_{role}s.txt").read_text(encoding="utf-8"))
            for role in ("question", "answer")}


class FakeLimiter:
    def __init__(self):
        self.calls = 0
        self.delays = []

    def acquire(self):
        self.calls += 1

    def defer(self, seconds):
        self.delays.append(seconds)


class Response:
    def __init__(self, text="", status=200, finish="stop", usage=True):
        self.status_code = status
        self.headers = {"Retry-After": "7"} if status == 429 else {}
        self.data = {"choices": [{"message": {"content": text}, "finish_reason": finish}]}
        if usage:
            self.data["usage"] = {"prompt_tokens": 100, "completion_tokens": 20}

    def json(self):
        return self.data


def client(root, monkeypatch, responses):
    root.mkdir(exist_ok=True)
    monkeypatch.setenv("APS_TEST_KEY", "fake-test-key")
    models = {role: qa.ModelSettings(role + "-model", "https://example.invalid/v1", "APS_TEST_KEY", 1000,
                                     "max_tokens", 1, 4) for role in ("question", "answer")}
    limiter = FakeLimiter()
    requests = []
    def post(url, **kwargs):
        requests.append(kwargs["json"])
        return responses.pop(0)
    monkeypatch.setattr(qa.requests, "post", post)
    return qa.ModelClient(models, limiter, qa.Journal(root), 1), requests


def test_no_six_question_cap_and_exact_duplicate_removal():
    questions = [f"Distinct question {i}?" for i in range(12)]
    assert qa.parse_questions({"questions": questions + [questions[0].upper()]}) == questions
    assert qa.parse_questions({"questions": []}) == []


def test_answers_match_ids_even_when_reordered_and_can_abstain():
    questions = [{"question_id": "q1"}, {"question_id": "q2"}]
    answers = qa.parse_answers({"answers": [{"question_id": "q2", "answer": None, "reason": "Not in source"},
                                            {"question_id": "q1", "answer": "Supported"}]}, questions)
    assert [a["question_id"] for a in answers] == ["q1", "q2"]
    assert answers[1]["answer"] is None
    for invalid in ([], [{"question_id": "q3", "answer": "Unknown"}],
                    [{"question_id": "q1", "answer": "A"}, {"question_id": "q1", "answer": "B"}]):
        with pytest.raises(ValueError):
            qa.parse_answers({"answers": invalid}, questions)


def test_preview_requires_no_key_or_model_and_never_calls_provider(tmp_path, monkeypatch, article):
    source = tmp_path / "input"
    source.mkdir()
    path = source / "articles.jsonl"
    rows = [{**article, "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, lang)), "lang": lang} for lang in qa.LANGUAGES]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    (source / "summary.json").write_text(json.dumps({"public_excerpt_uuids": [rows[0]["uuid"]]}), encoding="utf-8")
    root = tmp_path / "output"
    assert qa.main(["--input", str(path), "--output", str(root), "--per-language", "1"]) == 0
    preview = [json.loads(line) for line in (root / "prompt-preview.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(preview) == 3
    assert "public excerpt" in preview[0]["question_prompt"]
    assert "{{QUESTIONS_FROM_QUESTION_MODEL}}" in preview[0]["answer_prompt"]
    assert not (root / "usage.jsonl").exists()
    assert not (root / "qa-candidates.jsonl").exists()


def test_two_models_multiple_pairs_and_resume_without_rebilling(tmp_path, monkeypatch, article, templates):
    questions = [f"Question {i}?" for i in range(8)]
    identified = qa.identify_questions(article["uuid"], questions)
    responses = [Response(json.dumps({"questions": questions})), Response(json.dumps({"answers": [
        {"question_id": item["question_id"], "answer": "Answer to " + item["question"]}
        for item in reversed(identified)]}))]
    api, requests = client(tmp_path, monkeypatch, responses)
    result = qa.process_article(article, templates, set(), api, tmp_path, "run", 2)
    assert len(result["pairs"]) == 8
    assert [r["model"] for r in requests] == ["question-model", "answer-model"]
    assert article["text"] in json.loads(requests[1]["messages"][1]["content"].split("BEGIN_UNTRUSTED_SOURCE\n")[1].split("\nEND_UNTRUSTED_SOURCE")[0])
    for pair in result["pairs"]:
        assert set(pair) == set(qa.ARTICLE_FIELDS) | {"article_uuid", "instruction", "output"}
        assert pair["text"] == article["text"] and pair["article_uuid"] == article["uuid"]
        assert pair["uuid"] != article["uuid"]
        assert pair["output"] == "Answer to " + pair["instruction"]
        question = next(item for item in identified if item["question_id"] == pair["uuid"])
        assert pair["instruction"] == question["question"]
    assert result["questions"] == identified
    again = qa.process_article(article, templates, set(), api, tmp_path, "run", 2)
    assert again == result and len(requests) == 2
    assert qa.usage_report(tmp_path, api.models)["by_phase"]["question"]["input_tokens"] == 100


def test_truncation_is_not_accepted_usage_is_kept_and_question_phase_survives(tmp_path, monkeypatch, article, templates):
    responses = [Response('{"questions":["What happened?"]}'), Response('{"answers":[', finish="length")]
    api, requests = client(tmp_path, monkeypatch, responses)
    failed = qa.process_article(article, templates, set(), api, tmp_path, "run", 3)
    assert failed["phase"] == "answer" and failed["error"] == "output_truncated_increase_output_tokens"
    assert len(requests) == 2  # Do not spend retries repeating a truncated response.
    report = qa.usage_report(tmp_path, api.models)
    assert report["by_phase"]["answer"]["output_tokens"] == 20
    question_id = qa.identify_questions(article["uuid"], ["What happened?"])[0]["question_id"]
    state = json.loads((tmp_path / "checkpoints" / (article["uuid"] + ".json")).read_text(encoding="utf-8"))
    assert state["questions"][0]["question_id"] == question_id
    responses.append(Response(json.dumps({"answers": [{"question_id": question_id, "answer": "Event."}]})))
    resumed = qa.process_article(article, templates, set(), api, tmp_path, "run", 3)
    assert len(resumed["pairs"]) == 1
    assert [r["model"] for r in requests] == ["question-model", "answer-model", "answer-model"]


def test_429_and_invalid_json_retries_are_all_limited_and_accounted(tmp_path, monkeypatch):
    responses = [Response(status=429), Response("not JSON"), Response('{"questions":["Supported?"]}')]
    api, requests = client(tmp_path, monkeypatch, responses)
    assert qa.generate(api, "question", "test", "prompt", qa.parse_questions, 3) == ["Supported?"]
    assert api.limiter.calls == 3 and 7 in api.limiter.delays
    report = qa.usage_report(tmp_path, api.models)["by_phase"]["question"]
    assert report["attempts"] == 3 and report["input_tokens"] == 200 and report["output_tokens"] == 40
    assert report["attempts_without_complete_usage"] == 1
    assert report["known_usage_cost_estimate_usd"] == pytest.approx(0.00036)


def test_unknown_prices_are_null_not_free(tmp_path, monkeypatch):
    api, requests = client(tmp_path, monkeypatch, [Response('{"questions":[]}')])
    qa.generate(api, "question", "test", "prompt", qa.parse_questions, 1)
    models = {role: qa.ModelSettings(settings.model, settings.base_url, settings.key_env,
                                    settings.output_tokens, settings.token_parameter)
              for role, settings in api.models.items()}
    assert qa.usage_report(tmp_path, models)["known_usage_cost_estimate_usd"] is None


def test_bounded_runner_keeps_unanswered_out_of_candidates(tmp_path, monkeypatch, article, templates):
    questions = qa.identify_questions(article["uuid"], ["Question 1?", "Question 2?"])
    responses = [Response('{"questions":["Question 1?","Question 2?"]}'), Response(json.dumps({"answers": [
        {"question_id": questions[0]["question_id"], "answer": "Supported."},
        {"question_id": questions[1]["question_id"], "answer": None, "reason": "Not in text"}]}))]
    api, requests = client(tmp_path, monkeypatch, responses)
    args = type("Args", (), {"concurrency": 1, "max_attempts": 2})()
    summary = qa.run_articles([article], templates, set(), api, tmp_path, "run", args)
    assert summary["pairs"] == 1 and summary["unanswered"] == 1
    assert len((tmp_path / "qa-candidates.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    exported_questions = json.loads((tmp_path / "questions.jsonl").read_text(encoding="utf-8"))
    assert exported_questions["questions"] == questions
    unanswered = json.loads((tmp_path / "unanswered.jsonl").read_text(encoding="utf-8"))
    assert unanswered["question_id"] == questions[1]["question_id"]
    pair = json.loads((tmp_path / "qa-candidates.jsonl").read_text(encoding="utf-8"))
    assert pair["uuid"] == questions[0]["question_id"]
    assert not (tmp_path / "qa-candidates.jsonl.tmp").exists()


def test_limiter_enforces_one_shared_schedule(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(qa.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(qa.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    limiter = qa.RateLimiter(20)
    limiter.acquire()
    limiter.acquire()
    assert clock[0] == pytest.approx(3)
    limiter.defer(7)
    limiter.acquire()
    assert clock[0] == pytest.approx(10)


def test_cli_preview_then_live_resume_and_configuration_guard(tmp_path, monkeypatch, article):
    source = tmp_path / "input"
    source.mkdir()
    records = [{**article, "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, language)), "lang": language}
               for language in qa.LANGUAGES]
    path = source / "articles.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    root = tmp_path / "output"
    monkeypatch.setenv("APS_TEST_KEY", "fake-test-key")
    monkeypatch.setattr(qa, "RateLimiter", lambda rpm: FakeLimiter())
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs["json"]["model"])
        if calls[-1] == "question-model":
            return Response('{"questions":["First question?","Second question?"]}')
        prompt = kwargs["json"]["messages"][1]["content"]
        questions = json.loads(prompt.split("BEGIN_UNTRUSTED_QUESTIONS\n")[1].split("\nEND_UNTRUSTED_QUESTIONS")[0])
        return Response(json.dumps({"answers": [{"question_id": question["question_id"],
                                                 "answer": "Answer to " + question["question"]}
                                                for question in reversed(questions)]}))
    monkeypatch.setattr(qa.requests, "post", post)
    arguments = ["--input", str(path), "--output", str(root), "--all",
                 "--question-model", "question-model", "--answer-model", "answer-model",
                 "--question-key-env", "APS_TEST_KEY", "--answer-key-env", "APS_TEST_KEY"]
    assert qa.main(arguments) == 0
    assert calls == []
    assert qa.main(arguments + ["--run"]) == 0
    assert len(calls) == 6
    assert qa.main(arguments + ["--run"]) == 0
    assert len(calls) == 6
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["pairs"] == 6 and summary["completed_articles"] == 3
    assert summary["usage"]["known_usage_cost_estimate_usd"] is None
    changed = ["different-model" if value == "answer-model" else value for value in arguments]
    with pytest.raises(SystemExit):
        qa.main(changed + ["--run"])
    assert len(calls) == 6


def test_category_and_unsuitable_question_examples_are_rendered(article, templates):
    prompt = qa.render(templates["question"], article, set())
    assert prompt.startswith('You create questions for a multilingual training corpus in the category "Economy".')
    assert "${category}" not in prompt
    assert "illustrations, not an exhaustive list" in prompt
    assert "Return {\"questions\": []} only if none remain" in prompt
    assert "The presence of an unsuitable candidate, or a short source, is not by itself" in prompt


def test_question_ids_are_stable_across_order_and_distinct_across_articles(article):
    texts = ["Who announced the project?", "When will the project begin?"]
    original = qa.identify_questions(article["uuid"], texts)
    reordered = qa.identify_questions(article["uuid"], list(reversed(texts)))
    assert {q["question"]: q["question_id"] for q in original} == {
        q["question"]: q["question_id"] for q in reordered}
    other = qa.identify_questions(str(uuid.uuid5(uuid.NAMESPACE_URL, "different-article")), texts)
    assert {q["question_id"] for q in original}.isdisjoint(q["question_id"] for q in other)
    for item in original:
        assert str(uuid.UUID(item["question_id"])) == item["question_id"]
    assert qa.identify_questions(article["uuid"], [texts[0].upper()])[0]["question_id"] == original[0]["question_id"]


def test_unknown_answer_id_cannot_create_a_pair(tmp_path, monkeypatch, article, templates):
    responses = [Response('{"questions":["What happened?"]}'), Response(json.dumps({"answers": [
        {"question_id": str(uuid.uuid4()), "answer": "Answer with an unrelated ID."}]}))]
    api, requests = client(tmp_path, monkeypatch, responses)
    result = qa.process_article(article, templates, set(), api, tmp_path, "run", 1)
    assert result["error"] == "invalid_JSON_or_response_schema"
    assert result["phase"] == "answer" and "pairs" not in result
    assert len(requests) == 2


def test_corrupt_checkpoint_question_id_is_not_used(tmp_path, monkeypatch, article, templates):
    api, requests = client(tmp_path, monkeypatch, [])
    qa.write_json(tmp_path / "checkpoints" / (article["uuid"] + ".json"),
                  {"fingerprint": "run", "questions": [{"question_id": str(uuid.uuid4()), "question": "What happened?"}]})
    result = qa.process_article(article, templates, set(), api, tmp_path, "run", 1)
    assert result["error"] == "invalid_checkpoint_or_local_IO_error"
    assert requests == []


def test_no_useful_questions_skips_answer_call(tmp_path, monkeypatch, article, templates):
    api, requests = client(tmp_path, monkeypatch, [Response('{"questions":[]}')])
    result = qa.process_article(article, templates, set(), api, tmp_path, "run", 1)
    assert result["questions"] == [] and result["pairs"] == []
    assert [request["model"] for request in requests] == ["question-model"]
