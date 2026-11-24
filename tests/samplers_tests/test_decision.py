from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest

import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import DecisionSampler
from optuna.samplers import PartialFixedSampler
from optuna.samplers._decision._candidates import Candidate
from optuna.samplers._decision._candidates import generate_mixed_candidates
from optuna.samplers._decision._client import _HTTPStatusError
from optuna.samplers._decision._client import DecisionResponseError
from optuna.samplers._decision._client import DecisionTransportError
from optuna.samplers._decision._client import load_dotenv
from optuna.samplers._decision._client import OpenAIDecisionsClient
from optuna.samplers._decision._client import ScoreLevel
from optuna.samplers._decision._client import ScoreQuestion
from optuna.samplers._decision.sampler import _select_baseline
from optuna.samplers._decision.sampler import _select_rank_blend
from optuna.samplers._decision.sampler import _serialize_study
from optuna.trial import Trial


class MockDecisionClient:
    """Deterministic SHA-256 DecisionClient fake for offline tests."""

    def __init__(
        self,
        *,
        fail_with: Exception | None = None,
        refuse_names: set[str] | None = None,
    ) -> None:
        self._fail_with = fail_with
        self._refuse_names = set(refuse_names) if refuse_names else set()
        self.calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return "mock-decision-v1"

    def score(
        self,
        shared_input: str,
        questions: Sequence[ScoreQuestion],
        *,
        prompt_cache_key: str | None = None,
    ) -> list[float | None]:
        self.calls.append(
            {
                "shared_input": shared_input,
                "questions": tuple(questions),
                "prompt_cache_key": prompt_cache_key,
            }
        )
        if self._fail_with is not None:
            raise self._fail_with
        out: list[float | None] = []
        for q in questions:
            if q.name in self._refuse_names:
                out.append(None)
                continue
            digest = hashlib.sha256(f"{shared_input}|{q.instructions}".encode()).digest()
            val = int.from_bytes(digest[:8], "big") / float(1 << 64)
            out.append(val * (len(q.levels) - 1))
        return out


def _sample_space() -> dict[str, BaseDistribution]:
    return {
        "x_float": FloatDistribution(-5.0, 5.0),
        "x_log": FloatDistribution(1e-5, 1.0, log=True),
        "x_step": FloatDistribution(0.0, 1.0, step=0.25),
        "i_lin": IntDistribution(1, 10),
        "i_log": IntDistribution(2, 64, log=True),
        "i_step": IntDistribution(2, 10, step=2),
        "cat": CategoricalDistribution(("adam", "sgd", "rmsprop")),
    }


def _levels() -> tuple[ScoreLevel, ...]:
    return (
        ScoreLevel(label="Low", description="Unlikely to improve"),
        ScoreLevel(label="High", description="Likely to improve"),
    )


def test_openai_client_wire_format_chunking_and_retries() -> None:
    client = OpenAIDecisionsClient(api_key="test-key", max_retries=2, backoff_base=0.001)
    q = ScoreQuestion(name="s0", instructions="Rate candidate.", levels=_levels())
    payload = client._build_payload("history", [q], "cache_123")
    assert payload == {
        "model": "gpt-6-luna",
        "input": "history",
        "prompt_cache_key": "cache_123",
        "questions": [
            {
                "type": "score",
                "name": "s0",
                "instructions": "Rate candidate.",
                "levels": [
                    {"label": "Low", "description": "Unlikely to improve"},
                    {"label": "High", "description": "Likely to improve"},
                ],
            }
        ],
    }
    assert client._parse_scores(
        {"answers": [{"type": "score", "score": 0.8}, {"type": "refusal"}]}, 2
    ) == [0.8, None]

    # Auto-chunking over 400 questions preserves prompt_cache_key.
    many_q = [
        ScoreQuestion(name=f"q_{i}", instructions=f"c{i}", levels=_levels()) for i in range(450)
    ]
    seen: list[dict[str, Any]] = []

    def fake_post(url: str, headers: dict[str, str], body: bytes) -> dict[str, Any]:
        del url, headers
        req = json.loads(body.decode("utf-8"))
        seen.append(req)
        return {"answers": [{"type": "score", "score": 0.6} for _ in req["questions"]]}

    with patch.object(client, "_post_json", side_effect=fake_post):
        scores = client.score("shared", many_q, prompt_cache_key="same_key")
    assert [len(p["questions"]) for p in seen] == [400, 50]
    assert all(p["prompt_cache_key"] == "same_key" for p in seen)
    assert len(scores) == 450

    # Retries 429 and fails fast on 400.
    attempts = {"n": 0}

    def flaky(url: str, headers: dict[str, str], body: bytes) -> dict[str, Any]:
        del url, headers, body
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _HTTPStatusError(429, "rate limited")
        return {"answers": [{"type": "score", "score": 0.9}]}

    with patch.object(client, "_post_json", side_effect=flaky):
        assert client.score("in", [q]) == [0.9]
    assert attempts["n"] == 3
    with patch.object(client, "_post_json", side_effect=_HTTPStatusError(400, "bad request")):
        with pytest.raises(DecisionTransportError, match="HTTP 400"):
            client.score("in", [q])


def test_load_dotenv_and_api_key_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        '# comment\nOPENAI_API_KEY="sk-test-123"\nexport OTHER_VAR=hello\n', encoding="utf-8"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OTHER_VAR", raising=False)
    monkeypatch.chdir(tmp_path)

    loaded = load_dotenv()
    assert loaded == {"OPENAI_API_KEY": "sk-test-123", "OTHER_VAR": "hello"}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert OpenAIDecisionsClient()._resolve_api_key() == "sk-test-123"


def test_serializer_cache_key_and_non_finite_filtering() -> None:
    study = optuna.create_study(direction="minimize")
    space: dict[str, BaseDistribution] = {"x": FloatDistribution(-5.0, 5.0)}
    _, key0, best0, _ = _serialize_study(
        study,
        space,
        model="gpt-6-luna",
        task_description="Min x^2",
        param_descriptions={},
        state_formatter=None,
        max_history_trials=40,
    )
    assert len(key0) == 32 and best0 is None

    study.add_trial(optuna.trial.create_trial(params={"x": 2.0}, distributions=space, value=4.0))
    study.add_trial(
        optuna.trial.create_trial(params={"x": 1.0}, distributions=space, value=float("-inf"))
    )
    study.add_trial(
        optuna.trial.create_trial(
            params={"x": -1.0}, distributions=space, state=optuna.trial.TrialState.FAIL
        )
    )
    text, key1, best1, _ = _serialize_study(
        study,
        space,
        model="gpt-6-luna",
        task_description="Min x^2",
        param_descriptions={},
        state_formatter=None,
        max_history_trials=40,
    )
    assert key1 != key0
    assert best1 == pytest.approx(4.0)
    assert "1 failed" in text and "1 non-finite" in text


def test_mixed_candidates_rank_blend_and_refusals() -> None:
    space = _sample_space()
    study = optuna.create_study(direction="minimize")
    for i in range(8):
        study.add_trial(
            optuna.trial.create_trial(
                params={
                    "x_float": -4.0 + i,
                    "x_log": 10.0 ** (-4.0 + 0.4 * i),
                    "x_step": 0.25 * (i % 5),
                    "i_lin": i + 1,
                    "i_log": 2 ** ((i % 5) + 1),
                    "i_step": 2 + 2 * (i % 5),
                    "cat": ("adam", "sgd", "rmsprop")[i % 3],
                },
                distributions=space,
                value=float(i),
            )
        )
    pool = generate_mixed_candidates(study, space, 24, np.random.RandomState(42))
    assert len(study.trials) == 8  # Zero dummy trials created.
    assert len(pool) == 24
    assert all(c.baseline_score is not None and math.isfinite(c.baseline_score) for c in pool)
    for cand in pool:
        for name, dist in space.items():
            assert dist._contains(dist.to_internal_repr(cand.params[name]))

    cands = [Candidate({"x": 0.1}, 1.0), Candidate({"x": 0.9}, 5.0)]
    rng = np.random.RandomState(0)
    assert (
        _select_rank_blend(cands, [3.8, 0.5], rng, model_weight=0.0, epsilon_explore=0.0)
        == cands[1]
    )
    assert (
        _select_rank_blend(cands, [3.8, 0.5], rng, model_weight=1.0, epsilon_explore=0.0)
        == cands[0]
    )
    assert _select_baseline(cands, rng) == cands[1]


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_decision_sampler_end_to_end_fallback_and_refusal(
    direction: Literal["minimize", "maximize"],
) -> None:
    space = _sample_space()
    client = MockDecisionClient()
    sampler = DecisionSampler(
        client=client, n_candidates=16, n_startup_trials=0, search_space=space, seed=123
    )
    study = optuna.create_study(direction=direction, sampler=sampler)

    def objective(trial: Trial) -> float:
        x_float = trial.suggest_float("x_float", -5.0, 5.0)
        x_log = trial.suggest_float("x_log", 1e-5, 1.0, log=True)
        x_step = trial.suggest_float("x_step", 0.0, 1.0, step=0.25)
        i_lin = trial.suggest_int("i_lin", 1, 10)
        i_log = trial.suggest_int("i_log", 2, 64, log=True)
        i_step = trial.suggest_int("i_step", 2, 10, step=2)
        cat = trial.suggest_categorical("cat", ("adam", "sgd", "rmsprop"))
        val = -(x_float**2) - abs(math.log10(x_log) + 2.0) - x_step - 0.1 * i_lin - 0.01 * i_log
        return (val + (1.0 if i_step == 6 and cat == "adam" else 0.0)) * (
            1.0 if direction == "maximize" else -1.0
        )

    study.optimize(objective, n_trials=8)
    assert len(study.trials) == 8 and len(client.calls) == 8

    # Fallback vs raise on transport error or total refusal.
    small: dict[str, BaseDistribution] = {"x": FloatDistribution(-2.0, 2.0)}
    fb_study = optuna.create_study(
        sampler=DecisionSampler(
            client=MockDecisionClient(fail_with=DecisionTransportError("timeout")),
            n_candidates=4,
            n_startup_trials=0,
            search_space=small,
            on_error="fallback",
            seed=7,
        )
    )
    fb_study.optimize(lambda t: t.suggest_float("x", -2.0, 2.0) ** 2, n_trials=3)
    assert len(fb_study.trials) == 3

    with pytest.raises(DecisionTransportError, match="boom"):
        optuna.create_study(
            sampler=DecisionSampler(
                client=MockDecisionClient(fail_with=DecisionTransportError("boom")),
                n_candidates=4,
                n_startup_trials=0,
                search_space=small,
                on_error="raise",
                seed=7,
            )
        ).optimize(lambda t: t.suggest_float("x", -2.0, 2.0) ** 2, n_trials=1)

    with pytest.raises(DecisionResponseError, match="refused"):
        optuna.create_study(
            sampler=DecisionSampler(
                client=MockDecisionClient(
                    refuse_names={"score_0", "score_1", "score_2", "score_3"}
                ),
                n_candidates=4,
                n_startup_trials=0,
                search_space=small,
                on_error="raise",
                seed=7,
            )
        ).optimize(lambda t: t.suggest_float("x", -2.0, 2.0) ** 2, n_trials=1)


def test_partial_fixed_and_conditional_group_support() -> None:
    pf = PartialFixedSampler(
        {"x": 1.5},
        DecisionSampler(client=MockDecisionClient(), n_candidates=8, n_startup_trials=0, seed=42),
    )
    pf_study = optuna.create_study(sampler=pf)
    pf_study.optimize(
        lambda t: (
            t.suggest_float("x", -5.0, 5.0) ** 2
            + t.suggest_float("y", -5.0, 5.0) ** 2
            + t.suggest_float("z", 3.0, 3.0)
        ),
        n_trials=5,
    )
    assert all(t.params["x"] == 1.5 and t.params["z"] == 3.0 for t in pf_study.trials)

    client = MockDecisionClient()
    study = optuna.create_study(
        sampler=DecisionSampler(
            client=client,
            n_candidates=12,
            n_startup_trials=4,
            group=True,
            seed=42,
            warn_independent_sampling=False,
        )
    )

    def cond_obj(trial: Trial) -> float:
        booster = trial.suggest_categorical("booster", ("gbtree", "gblinear", "dart"))
        reg = trial.suggest_float("reg", 1e-4, 1.0, log=True)
        if booster in ("gbtree", "dart"):
            depth = trial.suggest_int("max_depth", 2, 8)
            eta = trial.suggest_float("eta", 1e-3, 0.5, log=True)
            loss = abs(eta - 0.1) + 0.02 * abs(depth - 4) + 0.01 * reg
            return loss + (
                0.05 * trial.suggest_float("rate_drop", 0.05, 0.5) if booster == "dart" else 0.0
            )
        return 0.5 + 0.1 * reg

    study.enqueue_trial({"booster": "gbtree", "reg": 0.1, "max_depth": 4, "eta": 0.1})
    study.enqueue_trial({"booster": "gblinear", "reg": 0.2})
    study.enqueue_trial(
        {"booster": "dart", "reg": 0.05, "max_depth": 5, "eta": 0.05, "rate_drop": 0.2}
    )
    study.optimize(cond_obj, n_trials=10)
    assert len(study.trials) == 10
    assert any(
        "Already selected in this trial:" in q.instructions
        for call in client.calls
        for q in call["questions"]
    )
    keys = [c["prompt_cache_key"] for c in client.calls]
    assert len(keys) > len(set(keys))
