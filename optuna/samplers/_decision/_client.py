from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any
from typing import Protocol
import urllib.error
import urllib.request


class DecisionClientError(RuntimeError):
    """Base error raised by a DecisionClient."""


class DecisionTransportError(DecisionClientError):
    """Raised on HTTP/network errors or timeouts."""


class DecisionResponseError(DecisionClientError):
    """Raised when a provider response is malformed or incompatible."""


@dataclass(frozen=True)
class ScoreLevel:
    label: str
    description: str | None = None


@dataclass(frozen=True)
class ScoreQuestion:
    name: str
    instructions: str
    levels: tuple[ScoreLevel, ...]

    def __post_init__(self) -> None:
        if not (2 <= len(self.levels) <= 10):
            raise ValueError(
                f"ScoreQuestion requires between 2 and 10 levels, got {len(self.levels)}."
            )


class DecisionClient(Protocol):
    @property
    def model(self) -> str: ...

    def score(
        self,
        shared_input: str,
        questions: Sequence[ScoreQuestion],
        *,
        prompt_cache_key: str | None = None,
    ) -> list[float | None]: ...


def load_dotenv(
    dotenv_path: str | Path | None = None, *, override: bool = False
) -> dict[str, str]:
    """Load environment variables from a ``.env`` file using only the Python stdlib."""
    target: Path | None = None
    if dotenv_path is not None:
        candidate = Path(dotenv_path)
        target = candidate if candidate.is_file() else None
    else:
        cwd = Path.cwd().resolve()
        for parent in (cwd, *cwd.parents):
            candidate = parent / ".env"
            if candidate.is_file():
                target = candidate
                break

    loaded: dict[str, str] = {}
    if target is None:
        return loaded
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, val = (part.strip() for part in line.split("=", 1))
        if not key:
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        loaded[key] = val
        if override or key not in os.environ:
            os.environ[key] = val
    return loaded


class _HTTPStatusError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class OpenAIDecisionsClient:
    """Zero-dependency HTTP client for OpenAI's Decisions API (``POST /v1/decisions``)."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-6-luna"
    MAX_QUESTIONS_LIMIT = 400
    RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        max_questions_per_request: int = MAX_QUESTIONS_LIMIT,
    ) -> None:
        if not (1 <= max_questions_per_request <= self.MAX_QUESTIONS_LIMIT):
            raise ValueError(
                f"max_questions_per_request must be in [1, {self.MAX_QUESTIONS_LIMIT}], "
                f"got {max_questions_per_request}."
            )
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}.")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}.")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._max_questions_per_request = max_questions_per_request

    @property
    def model(self) -> str:
        return self._model

    def _resolve_api_key(self) -> str:
        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            load_dotenv()
            key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise DecisionClientError(
                "OPENAI_API_KEY is not set. Pass `api_key=...`, set the OPENAI_API_KEY "
                "environment variable, or add it to a `.env` file."
            )
        return key

    def _build_payload(
        self,
        shared_input: str,
        questions: Sequence[ScoreQuestion],
        prompt_cache_key: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "input": shared_input,
            "questions": [
                {
                    "type": "score",
                    "name": q.name,
                    "instructions": q.instructions,
                    "levels": [
                        {"label": lvl.label}
                        if lvl.description is None
                        else {"label": lvl.label, "description": lvl.description}
                        for lvl in q.levels
                    ],
                }
                for q in questions
            ],
        }
        if prompt_cache_key is not None:
            if len(prompt_cache_key) > 64:
                raise ValueError(f"prompt_cache_key exceeds 64 chars: {len(prompt_cache_key)}.")
            payload["prompt_cache_key"] = prompt_cache_key
        return payload

    def _parse_scores(self, payload: dict[str, Any], n_questions: int) -> list[float | None]:
        raw_answers = payload.get("answers")
        if not isinstance(raw_answers, list) or len(raw_answers) != n_questions:
            raise DecisionResponseError(
                f"Expected {n_questions} answers in response, got {raw_answers!r}."
            )
        scores: list[float | None] = []
        for ans in raw_answers:
            if not isinstance(ans, dict):
                raise DecisionResponseError(f"Malformed answer entry: {ans!r}")
            ans_type = ans.get("type")
            if ans_type == "refusal":
                scores.append(None)
            elif ans_type == "score" and "score" in ans:
                val = float(ans["score"])
                if not math.isfinite(val):
                    raise DecisionResponseError(f"Non-finite score: {val}")
                scores.append(val)
            else:
                raise DecisionResponseError(f"Unexpected answer entry: {ans!r}")
        return scores

    def _post_json(self, url: str, headers: dict[str, str], body: bytes) -> dict[str, Any]:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise _HTTPStatusError(exc.code, err_body) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DecisionTransportError(f"HTTP transport error: {exc}") from exc

        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:
            raise DecisionResponseError(f"Invalid JSON response: {exc}") from exc
        if not isinstance(parsed, dict):
            raise DecisionResponseError("Response root must be a JSON object.")
        return parsed

    def _send_chunk(
        self,
        shared_input: str,
        questions: Sequence[ScoreQuestion],
        prompt_cache_key: str | None,
    ) -> list[float | None]:
        url = f"{self._base_url}/decisions"
        body = json.dumps(
            self._build_payload(shared_input, questions, prompt_cache_key),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._resolve_api_key()}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._post_json(url, headers, body)
                return self._parse_scores(resp, len(questions))
            except _HTTPStatusError as exc:
                last_error = DecisionTransportError(
                    f"HTTP {exc.status_code} from {url}: {exc.body[:500]}"
                )
                if exc.status_code not in self.RETRYABLE_STATUS_CODES:
                    raise last_error from exc
            except DecisionTransportError as exc:
                last_error = exc
            if attempt < self._max_retries:
                time.sleep(self._backoff_base * (2**attempt) * (0.75 + 0.5 * random.random()))
        assert last_error is not None
        raise last_error

    def score(
        self,
        shared_input: str,
        questions: Sequence[ScoreQuestion],
        *,
        prompt_cache_key: str | None = None,
    ) -> list[float | None]:
        if not questions:
            raise ValueError("questions must contain at least 1 question.")
        chunk_size = self._max_questions_per_request
        scores: list[float | None] = []
        for start in range(0, len(questions), chunk_size):
            scores.extend(
                self._send_chunk(
                    shared_input, questions[start : start + chunk_size], prompt_cache_key
                )
            )
        return scores
