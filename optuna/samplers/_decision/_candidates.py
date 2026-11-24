from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna._transform import _SearchSpaceTransform
from optuna.distributions import CategoricalDistribution
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters
from optuna.samplers._tpe.sampler import _split_trials
from optuna.samplers._tpe.sampler import default_gamma
from optuna.samplers._tpe.sampler import default_weights
from optuna.study import StudyDirection
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


_TPE_PARAMS = _ParzenEstimatorParameters(
    prior_weight=1.0,
    consider_magic_clip=True,
    consider_endpoints=False,
    weights=default_weights,
    multivariate=True,
)


@dataclass(frozen=True)
class Candidate:
    params: dict[str, Any]
    baseline_score: float | None = None


def _canonical_key(
    params: dict[str, Any], search_space: dict[str, BaseDistribution]
) -> tuple[Any, ...]:
    parts: list[Any] = []
    for name in sorted(search_space):
        dist = search_space[name]
        val = params[name]
        if isinstance(dist, CategoricalDistribution):
            parts.append((name, int(dist.to_internal_repr(val))))
        elif isinstance(val, (int, np.integer)) and not isinstance(val, bool):
            parts.append((name, int(val)))
        elif isinstance(val, (float, np.floating)):
            parts.append((name, round(float(val), 12)))
        else:
            parts.append((name, repr(val)))
    return tuple(parts)


def _to_native_params(
    params: dict[str, Any], search_space: dict[str, BaseDistribution]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, dist in search_space.items():
        val = params[name]
        if isinstance(dist, CategoricalDistribution):
            out[name] = val
        elif isinstance(val, np.integer):
            out[name] = int(val)
        elif isinstance(val, np.floating):
            out[name] = float(val)
        else:
            out[name] = val
    return out


def _evaluated_keys(
    study: Study, search_space: dict[str, BaseDistribution]
) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    space_keys = set(search_space)
    for t in study._get_trials(deepcopy=False, use_cache=False):
        if t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING) and (
            space_keys <= t.params.keys()
        ):
            try:
                keys.add(_canonical_key(t.params, search_space))
            except Exception:
                continue
    return keys


def _fit_tpe(
    study: Study, search_space: dict[str, BaseDistribution], n_startup_trials: int
) -> tuple[_ParzenEstimator, _ParzenEstimator] | None:
    trials = study._get_trials(
        deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED), use_cache=False
    )
    space_keys = set(search_space)
    matching = [t for t in trials if space_keys <= t.params.keys()]
    if len(matching) < n_startup_trials or not matching:
        return None
    below, above = _split_trials(study, matching, default_gamma(len(matching)))

    def _internal(ts: list[FrozenTrial]) -> dict[str, np.ndarray]:
        return {
            name: np.asarray([dist.to_internal_repr(t.params[name]) for t in ts], dtype=np.float64)
            for name, dist in search_space.items()
        }

    return (
        _ParzenEstimator(_internal(below), search_space, _TPE_PARAMS),
        _ParzenEstimator(_internal(above), search_space, _TPE_PARAMS),
    )


def generate_mixed_candidates(
    study: Study,
    search_space: dict[str, BaseDistribution],
    n_candidates: int,
    rng: np.random.RandomState,
    *,
    tpe_fraction: float = 0.5,
    local_fraction: float = 0.25,
    tpe_startup_trials: int = 5,
    local_top_k: int = 3,
    local_sigma: float = 0.12,
    categorical_flip_prob: float = 0.2,
) -> list[Candidate]:
    """Generate valid candidates combining TPE, local incumbent perturbations, and random."""
    if n_candidates <= 0 or not search_space:
        return []

    seen = _evaluated_keys(study, search_space)
    transform = _SearchSpaceTransform(search_space, transform_0_1=True)
    dim = int(transform.bounds.shape[0])
    fitted = _fit_tpe(study, search_space, tpe_startup_trials)
    raw_params: list[dict[str, Any]] = []

    def _try_add(params: dict[str, Any], limit: int) -> bool:
        if len(raw_params) >= limit:
            return False
        key = _canonical_key(params, search_space)
        if key in seen:
            return False
        seen.add(key)
        raw_params.append(params)
        return True

    # 1. TPE proposals (top log l(x) - log g(x) first).
    n_tpe = int(round(n_candidates * tpe_fraction)) if fitted is not None else 0
    if fitted is not None and n_tpe > 0:
        mpe_below, mpe_above = fitted
        samples = mpe_below.sample(rng, max(n_tpe * 3, n_tpe))
        acq = mpe_below.log_pdf(samples) - mpe_above.log_pdf(samples)
        for idx in np.argsort(-acq):
            p = _to_native_params(
                {k: d.to_external_repr(samples[k][idx].item()) for k, d in search_space.items()},
                search_space,
            )
            _try_add(p, n_tpe)
            if len(raw_params) >= n_tpe:
                break

    # 2. Local perturbations around top-k completed incumbents.
    n_local = int(round(n_candidates * local_fraction))
    if n_local > 0:
        space_keys = set(search_space)
        finite = [
            t
            for t in study._get_trials(
                deepcopy=False, states=(TrialState.COMPLETE,), use_cache=False
            )
            if t.value is not None
            and math.isfinite(float(t.value))
            and space_keys <= t.params.keys()
        ]
        if finite:
            top = sorted(
                finite,
                key=lambda t: float(t.value) if t.value is not None else 0.0,
                reverse=(study.direction == StudyDirection.MAXIMIZE),
            )[: max(1, local_top_k)]
            anchors = [transform.transform(t.params) for t in top]
            cat_groups = [
                cols
                for d, cols in zip(
                    search_space.values(), transform.column_to_encoded_columns, strict=True
                )
                if isinstance(d, CategoricalDistribution)
            ]
            num_cols = [
                int(cols[0])
                for d, cols in zip(
                    search_space.values(), transform.column_to_encoded_columns, strict=True
                )
                if not isinstance(d, CategoricalDistribution)
            ]
            local_limit = min(n_candidates, len(raw_params) + n_local)
            for _ in range(max(n_local * 8, 24)):
                if len(raw_params) >= local_limit:
                    break
                anchor = anchors[int(rng.randint(0, len(anchors)))].copy()
                if num_cols:
                    anchor[num_cols] = np.clip(
                        anchor[num_cols] + rng.normal(0.0, local_sigma, size=len(num_cols)),
                        0.0,
                        1.0,
                    )
                for cols in cat_groups:
                    if len(cols) > 1 and float(rng.uniform()) < categorical_flip_prob:
                        anchor[cols] = 0.0
                        anchor[cols[int(rng.randint(0, len(cols)))]] = 1.0
                _try_add(
                    _to_native_params(transform.untransform(anchor), search_space), local_limit
                )

    # 3. Global uniform random fill (plus non-dedup fallback if tiny space is exhausted).
    max_draws = max(n_candidates * 8, n_candidates + 16)
    drawn = 0
    while len(raw_params) < n_candidates and drawn < max_draws:
        batch = min(max(n_candidates, 16), max_draws - drawn)
        drawn += batch
        for row in rng.uniform(0.0, 1.0, size=(batch, dim)):
            _try_add(_to_native_params(transform.untransform(row), search_space), n_candidates)
            if len(raw_params) >= n_candidates:
                break
    if not raw_params:
        raw_params.append(
            _to_native_params(transform.untransform(rng.uniform(0.0, 1.0, size=dim)), search_space)
        )

    # Assign unified TPE baseline scores across the full pool when TPE is warmed up.
    scores: list[float | None] = [None] * len(raw_params)
    if fitted is not None:
        mpe_below, mpe_above = fitted
        internal_batch = {
            name: np.asarray(
                [dist.to_internal_repr(p[name]) for p in raw_params], dtype=np.float64
            )
            for name, dist in search_space.items()
        }
        acq_all = mpe_below.log_pdf(internal_batch) - mpe_above.log_pdf(internal_batch)
        scores = [float(v) if math.isfinite(float(v)) else None for v in acq_all]

    return [
        Candidate(params=p, baseline_score=s)
        for p, s in zip(raw_params[:n_candidates], scores, strict=True)
    ]
