from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import hashlib
import logging
import math
from typing import Any
from typing import Literal
from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers._base import _INDEPENDENT_SAMPLING_WARNING_TEMPLATE
from optuna.samplers._base import BaseSampler
from optuna.samplers._decision._candidates import Candidate
from optuna.samplers._decision._candidates import generate_mixed_candidates
from optuna.samplers._decision._client import DecisionClient
from optuna.samplers._decision._client import DecisionResponseError
from optuna.samplers._decision._client import OpenAIDecisionsClient
from optuna.samplers._decision._client import ScoreLevel
from optuna.samplers._decision._client import ScoreQuestion
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers._random import RandomSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.search_space.group_decomposed import _GroupDecomposedSearchSpace
from optuna.search_space.group_decomposed import _SearchSpaceGroup
from optuna.study import StudyDirection
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


_logger = logging.getLogger(__name__)
_PROMPT_VERSION = "decision-v1"

DEFAULT_SCORE_LEVELS: tuple[ScoreLevel, ...] = (
    ScoreLevel(label="Very poor", description="Likely much worse than the current incumbent."),
    ScoreLevel(label="Poor", description="Unlikely to improve on the current incumbent."),
    ScoreLevel(label="Uncertain", description="Untested region with plausible upside."),
    ScoreLevel(
        label="Promising",
        description="Likely competitive with or better than the current incumbent.",
    ),
    ScoreLevel(label="Strongest", description="High chance of meaningful improvement."),
)


def _fmt_num(value: Any) -> str:
    if isinstance(value, (bool, int)):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if value == 0.0:
            return "0"
        return f"{value:.4e}" if (abs(value) < 1e-3 or abs(value) >= 1e5) else f"{value:.6g}"
    return repr(value)


def _fmt_params(params: dict[str, Any], search_space: dict[str, BaseDistribution]) -> str:
    keys = sorted(search_space) if search_space else sorted(params)
    return "{" + ", ".join(f"{k}={_fmt_num(params[k])}" for k in keys if k in params) + "}"


def _describe_dist(name: str, dist: BaseDistribution, desc: str | None) -> str:
    if isinstance(dist, (FloatDistribution, IntDistribution)):
        dtype = "float" if isinstance(dist, FloatDistribution) else "int"
        extras = (["log-scale"] if dist.log else []) + (
            [f"step={_fmt_num(dist.step)}"]
            if (dist.step is not None and (dtype == "float" or dist.step != 1))
            else []
        )
        suffix = f" ({', '.join(extras)})" if extras else ""
        spec = f"{dtype} in [{_fmt_num(dist.low)}, {_fmt_num(dist.high)}]{suffix}"
    elif isinstance(dist, CategoricalDistribution):
        spec = f"categorical in {list(dist.choices)!r}"
    else:
        spec = type(dist).__name__
    return f"- {name}: {spec} -- {desc}" if desc else f"- {name}: {spec}"


def _serialize_study(
    study: Study,
    search_space: dict[str, BaseDistribution],
    *,
    model: str,
    task_description: str | None,
    param_descriptions: dict[str, str],
    state_formatter: Callable[[Study, dict[str, BaseDistribution], list[FrozenTrial]], str] | None,
    max_history_trials: int,
) -> tuple[str, str, float | None, str]:
    all_trials = study._get_trials(deepcopy=False, use_cache=False)
    completed = [t for t in all_trials if t.state == TrialState.COMPLETE]
    finite = [t for t in completed if t.value is not None and math.isfinite(float(t.value))]
    direction_str = "maximize" if study.direction == StudyDirection.MAXIMIZE else "minimize"

    sorted_by_val = sorted(
        finite,
        key=lambda t: float(t.value) if t.value is not None else 0.0,
        reverse=(study.direction == StudyDirection.MAXIMIZE),
    )
    best_trial = sorted_by_val[0] if sorted_by_val else None
    best_value = float(best_trial.value) if best_trial and best_trial.value is not None else None

    selected: list[FrozenTrial] = []
    if max_history_trials > 0 and finite:
        n_top = min(20, max_history_trials)
        picked: dict[int, FrozenTrial] = {t.number: t for t in sorted_by_val[:n_top]}
        top_nums = set(picked)
        for t in sorted(finite, key=lambda t: t.number, reverse=True)[:20]:
            if len(picked) >= max_history_trials:
                break
            picked.setdefault(t.number, t)
        selected = [t for t in sorted_by_val if t.number in top_nums] + sorted(
            (t for num, t in picked.items() if num not in top_nums), key=lambda t: t.number
        )

    if state_formatter is not None:
        shared_input = state_formatter(study, search_space, selected)
    else:
        lines = [
            "You are guiding sequential black-box hyperparameter optimization.",
            f"Objective direction: {direction_str} the objective value.",
        ]
        if task_description:
            lines.extend(["", "Task description:", task_description.strip()])
        lines.extend(["", "Search space:"])
        for key in sorted(search_space):
            lines.append(_describe_dist(key, search_space[key], param_descriptions.get(key)))
        lines.append("")
        if best_trial is not None and best_value is not None:
            lines.append(
                f"Current incumbent (best of {len(finite)} completed trials): "
                f"trial #{best_trial.number}, value={_fmt_num(best_value)}, "
                f"params={_fmt_params(best_trial.params, search_space)}"
            )
        else:
            lines.append("No completed trials yet (initial exploration phase).")
        notes = [
            f"{cnt} {lbl}"
            for cnt, lbl in (
                (sum(t.state == TrialState.FAIL for t in all_trials), "failed"),
                (len(completed) - len(finite), "non-finite"),
                (sum(t.state == TrialState.PRUNED for t in all_trials), "pruned"),
            )
            if cnt > 0
        ]
        if notes:
            lines.append("Other trials: " + ", ".join(notes) + ".")
        if selected:
            lines.extend(
                [
                    "",
                    f"Measured trial history ({len(selected)} shown out of "
                    f"{len(finite)} finite completed trials):",
                ]
            )
            for t in selected:
                val_s = _fmt_num(float(t.value)) if t.value is not None else "None"
                lines.append(
                    f"- trial #{t.number}: value={val_s}, "
                    f"params={_fmt_params(t.params, search_space)}"
                )
        shared_input = "\n".join(lines)

    cache_key = hashlib.sha256(f"{model}\n{_PROMPT_VERSION}\n{shared_input}".encode()).hexdigest()[
        :32
    ]
    return shared_input, cache_key, best_value, direction_str


def _select_rank_blend(
    candidates: Sequence[Candidate],
    model_scores: Sequence[float],
    rng: np.random.RandomState,
    *,
    model_weight: float,
    epsilon_explore: float,
) -> Candidate:
    n = len(candidates)
    if n == 1:
        return candidates[0]
    if epsilon_explore > 0.0 and float(rng.uniform()) < epsilon_explore:
        return candidates[int(rng.randint(0, n))]

    denom = float(n - 1)
    model_order = sorted(range(n), key=lambda i: (-model_scores[i], i))
    blended = np.zeros(n, dtype=np.float64)
    for pos, idx in enumerate(model_order):
        blended[idx] = 1.0 - (pos / denom)

    if all(c.baseline_score is not None and math.isfinite(c.baseline_score) for c in candidates):
        base_order = sorted(range(n), key=lambda i: (-(candidates[i].baseline_score or 0.0), i))
        base_norm = np.zeros(n, dtype=np.float64)
        for pos, idx in enumerate(base_order):
            base_norm[idx] = 1.0 - (pos / denom)
        blended = model_weight * blended + (1.0 - model_weight) * base_norm

    best_val = float(np.max(blended))
    tied = [i for i in range(n) if blended[i] == best_val]
    return candidates[tied[0] if len(tied) == 1 else tied[int(rng.randint(0, len(tied)))]]


def _select_baseline(candidates: Sequence[Candidate], rng: np.random.RandomState) -> Candidate:
    scored = [
        (i, float(c.baseline_score))
        for i, c in enumerate(candidates)
        if c.baseline_score is not None and math.isfinite(c.baseline_score)
    ]
    if scored:
        return candidates[max(scored, key=lambda kv: kv[1])[0]]
    return candidates[int(rng.randint(0, len(candidates)))]


@experimental_class("5.2.0")
class DecisionSampler(BaseSampler):
    """Single-objective sampler guided by a System One / Decision API model.

    Generates valid candidates via mixed TPE + local + random proposals, scores them with
    Decision API ordinal ``score`` questions, and selects via rank blending with epsilon
    exploration. Supports both fixed search spaces (including zero-shot startup with
    ``n_startup_trials=0`` and upfront ``search_space``) and conditional spaces (``group=True``).
    """

    def __init__(
        self,
        client: DecisionClient | None = None,
        *,
        n_candidates: int = 32,
        n_startup_trials: int = 5,
        search_space: dict[str, BaseDistribution] | None = None,
        group: bool = False,
        task_description: str | None = None,
        param_descriptions: dict[str, str] | None = None,
        state_formatter: Callable[[Study, dict[str, BaseDistribution], list[FrozenTrial]], str]
        | None = None,
        max_history_trials: int = 40,
        score_levels: Sequence[ScoreLevel] = DEFAULT_SCORE_LEVELS,
        model_weight: float = 0.7,
        epsilon_explore: float = 0.1,
        on_error: Literal["fallback", "raise"] = "fallback",
        independent_sampler: BaseSampler | None = None,
        seed: int | None = None,
        warn_independent_sampling: bool = True,
    ) -> None:
        if n_candidates < 1:
            raise ValueError(f"n_candidates must be >= 1, got {n_candidates}.")
        if n_startup_trials < 0:
            raise ValueError(f"n_startup_trials must be >= 0, got {n_startup_trials}.")
        if not (2 <= len(score_levels) <= 10):
            raise ValueError(f"score_levels must have 2..10 items, got {len(score_levels)}.")
        if not (0.0 <= model_weight <= 1.0):
            raise ValueError(f"model_weight must be in [0, 1], got {model_weight}.")
        if not (0.0 <= epsilon_explore <= 1.0):
            raise ValueError(f"epsilon_explore must be in [0, 1], got {epsilon_explore}.")
        if on_error not in ("fallback", "raise"):
            raise ValueError(f"on_error must be 'fallback' or 'raise', got {on_error!r}.")

        self._client: DecisionClient = client if client is not None else OpenAIDecisionsClient()
        self._n_candidates = n_candidates
        self._n_startup_trials = n_startup_trials
        self._explicit_search_space = dict(search_space) if search_space is not None else None
        self._group = group
        self._task_description = task_description
        self._param_descriptions = dict(param_descriptions) if param_descriptions else {}
        self._state_formatter = state_formatter
        self._max_history_trials = max_history_trials
        self._score_levels = tuple(score_levels)
        self._model_weight = model_weight
        self._epsilon_explore = epsilon_explore
        self._on_error: Literal["fallback", "raise"] = on_error
        self._independent_sampler: BaseSampler = independent_sampler or RandomSampler(seed=seed)
        self._rng = LazyRandomState(seed)
        self._search_space = IntersectionSearchSpace()
        self._group_decomposed = _GroupDecomposedSearchSpace() if group else None
        self._search_space_group: _SearchSpaceGroup | None = None
        self._warn_independent_sampling = warn_independent_sampling

    def reseed_rng(self) -> None:
        self._rng.rng.seed()
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        del trial
        if self._group:
            assert self._group_decomposed is not None
            self._search_space_group = self._group_decomposed.calculate(study, use_cache=False)
            if not self._search_space_group.search_spaces and self._explicit_search_space:
                return {
                    k: d for k, d in sorted(self._explicit_search_space.items()) if not d.single()
                }
            return {
                k: d
                for sub in self._search_space_group.search_spaces
                for k, d in sorted(sub.items())
                if not d.single()
            }
        raw = (
            self._explicit_search_space
            if self._explicit_search_space is not None
            else self._search_space.calculate(study)
        )
        return {k: d for k, d in raw.items() if not d.single()}

    def _sample_space(
        self,
        study: Study,
        trial: FrozenTrial,
        active_space: dict[str, BaseDistribution],
        *,
        selected_group_params: dict[str, Any] | None = None,
        shared_context_space: dict[str, BaseDistribution] | None = None,
    ) -> dict[str, Any]:
        candidates = generate_mixed_candidates(
            study, active_space, self._n_candidates, self._rng.rng
        )
        if not candidates:
            return {}
        if len(candidates) == 1:
            return dict(candidates[0].params)

        shared_input, cache_key, best_val, direction = _serialize_study(
            study,
            shared_context_space if shared_context_space is not None else active_space,
            model=self._client.model,
            task_description=self._task_description,
            param_descriptions=self._param_descriptions,
            state_formatter=self._state_formatter,
            max_history_trials=self._max_history_trials,
        )
        prefix = (
            "Already selected in this trial: {"
            + ", ".join(f"{k}={v!r}" for k, v in sorted(selected_group_params.items()))
            + "}. "
            if selected_group_params
            else ""
        )
        questions = [
            ScoreQuestion(
                name=f"score_{idx}",
                instructions=(
                    f"{prefix}Candidate configuration: {_fmt_params(c.params, active_space)}. "
                    + (
                        f"Rate how promising this initial configuration is for the task "
                        f"(goal: {direction})."
                        if best_val is None
                        else f"Rate how promising evaluating this candidate is compared to the "
                        f"current incumbent ({best_val:.6g}, goal: {direction})."
                    )
                ),
                levels=self._score_levels,
            )
            for idx, c in enumerate(candidates)
        ]

        try:
            raw_scores = self._client.score(shared_input, questions, prompt_cache_key=cache_key)
            if len(raw_scores) != len(candidates):
                raise DecisionResponseError(
                    f"Expected {len(candidates)} scores, got {len(raw_scores)}."
                )
            if all(s is None for s in raw_scores):
                raise DecisionResponseError("All score questions were refused by the model.")
            midpoint = (len(self._score_levels) - 1) / 2.0
            scores = [midpoint if s is None else float(s) for s in raw_scores]
            chosen = _select_rank_blend(
                candidates,
                scores,
                self._rng.rng,
                model_weight=self._model_weight,
                epsilon_explore=self._epsilon_explore,
            )
        except Exception as exc:
            if self._on_error == "raise":
                raise
            _logger.warning(
                "DecisionSampler falling back to baseline selection on Trial#%d due to %s: %s",
                trial.number,
                type(exc).__name__,
                exc,
            )
            chosen = _select_baseline(candidates, self._rng.rng)
        return dict(chosen.params)

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        self._raise_error_if_multi_objective(study)
        active = {k: d for k, d in search_space.items() if not d.single()}
        if not active:
            return {}
        completed = study._get_trials(
            deepcopy=False, states=(TrialState.COMPLETE,), use_cache=False
        )
        if len(completed) < self._n_startup_trials:
            return {}

        if (
            self._group
            and self._search_space_group is not None
            and len(self._search_space_group.search_spaces) > 1
        ):
            sub_spaces = [
                {k: d for k, d in sorted(sub.items()) if not d.single() and k in active}
                for sub in self._search_space_group.search_spaces
            ]
            sub_spaces = [s for s in sub_spaces if s]
            sub_spaces.sort(
                key=lambda sp: (
                    -sum(1 for t in completed if set(sp) <= t.params.keys()),
                    tuple(sorted(sp)),
                )
            )
            combined: dict[str, Any] = {}
            for sub in sub_spaces:
                combined.update(
                    self._sample_space(
                        study,
                        trial,
                        sub,
                        selected_group_params=combined,
                        shared_context_space=active,
                    )
                )
            return combined

        return self._sample_space(study, trial, active)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        self._raise_error_if_multi_objective(study)
        if self._warn_independent_sampling:
            completed = study._get_trials(
                deepcopy=False, states=(TrialState.COMPLETE,), use_cache=False
            )
            if len(completed) >= self._n_startup_trials:
                reason = (
                    "a newly discovered conditional parameter was not yet seen in prior trials"
                    if self._group
                    else "dynamic search space is not supported (set `group=True`)"
                )
                _logger.warning(
                    _INDEPENDENT_SAMPLING_WARNING_TEMPLATE.format(
                        param_name=param_name,
                        trial_number=trial.number,
                        independent_sampler_name=self._independent_sampler.__class__.__name__,
                        sampler_name=self.__class__.__name__,
                        fallback_reason=reason,
                    )
                )
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)
