"""One function, complete_json(), that every step uses to call a model and get JSON back.

Claude goes through the anthropic SDK (or Bedrock), OpenAI models through the openai SDK.
Each call is logged to runs/usage.jsonl so we can report cost and time."""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .config import RUNS, SETTINGS

_lock = threading.Lock()
_override: Optional[Callable[..., Any]] = None   # tests inject a fake here
_clients: dict[str, Any] = {}


class LLMError(RuntimeError):
    pass


@dataclass
class Call:
    role: str
    system: str
    user: str
    images: Optional[list[Path]] = None
    max_tokens: int = 16_000


def set_override(fn: Optional[Callable[[Call], Any]]) -> None:
    """Route all calls to ``fn(call) -> dict`` (used by offline tests)."""
    global _override
    _override = fn


def complete_json(role: str, system: str, user: str, images: Optional[list[Path]] = None,
                  max_tokens: int = 16_000) -> Any:
    """Return parsed JSON from the model assigned to ``role``. One repair retry on bad JSON."""
    call = Call(role, system, user, images, max_tokens)
    if _override is not None:
        return _override(call)
    spec = SETTINGS.model(role)
    provider, _, model = spec.partition(":")
    text = _dispatch(provider, model, call)
    try:
        return parse_json(text)
    except ValueError:
        repair = Call(role, system, user + "\n\nYour previous reply was not valid JSON. "
                      "Reply again with ONLY the JSON value, no prose.", images, max_tokens)
        return parse_json(_dispatch(provider, model, repair))


def parse_json(text: str) -> Any:
    """Parse the first JSON object/array in ``text`` (tolerates code fences and prose)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("No valid JSON in model output")


class BudgetExceeded(LLMError):
    pass


_spent = 0.0


def spent() -> float:
    """Estimated dollars spent by this process so far."""
    return _spent


def _check_budget() -> None:
    cap = os.getenv("AGENT_MAX_USD")
    if cap and _spent >= float(cap):
        raise BudgetExceeded(f"Spending cap reached (~${_spent:.2f} of ${float(cap):.2f}). "
                             "Raise AGENT_MAX_USD to continue.")


def _dispatch(provider: str, model: str, call: Call) -> str:
    _check_budget()
    t0 = time.time()
    if provider in ("anthropic", "bedrock"):
        text, usage = _anthropic(model, call, provider)
    elif provider == "openai":
        text, usage = _openai(model, call)
    elif provider == "openrouter":
        text, usage = _openrouter(model, call)
    else:
        raise LLMError(f"Unknown provider '{provider}'")
    _log(call.role, f"{provider}:{model}", usage, time.time() - t0)
    return text


def _client(provider: str):
    with _lock:
        if provider not in _clients:
            if provider == "anthropic":
                import anthropic
                _clients[provider] = anthropic.Anthropic(max_retries=4)
            elif provider == "bedrock":            # Claude on AWS Bedrock; uses the normal AWS credentials
                import anthropic
                _clients[provider] = anthropic.AnthropicBedrock(max_retries=4)
            elif provider == "openrouter":         # one key for Claude, GPT and Cohere (OpenAI-compatible API)
                import openai
                key = os.getenv("OPENROUTER_API_KEY", "").strip()
                if not key:
                    raise LLMError("OPENROUTER_API_KEY is not set (add it to .env)")
                _clients[provider] = openai.OpenAI(base_url=OPENROUTER_URL, api_key=key, max_retries=4,
                                                   timeout=600)
            else:
                import openai
                _clients[provider] = openai.OpenAI(max_retries=4)
        return _clients[provider]


def _anthropic(model: str, call: Call, provider: str = "anthropic") -> tuple[str, dict]:
    import anthropic

    content: list[dict] = []
    for img in call.images or []:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": base64.standard_b64encode(Path(img).read_bytes()).decode("utf-8")},
        })
    content.append({"type": "text", "text": call.user})
    client = _client(provider)
    kwargs: dict[str, Any] = dict(model=model, max_tokens=call.max_tokens, system=call.system,
                                  messages=[{"role": "user", "content": content}])
    if SETTINGS.effort and _supports_effort(model):
        kwargs["output_config"] = {"effort": SETTINGS.effort}
    try:
        # stream so long outputs (thinking + JSON) don't hit request timeouts
        with client.messages.stream(**kwargs) as stream:
            resp = stream.get_final_message()
    except anthropic.BadRequestError as exc:
        raise LLMError(f"Anthropic rejected the request ({model}): {exc}") from exc
    if resp.stop_reason == "refusal":
        raise LLMError(f"{model} declined the request (stop_reason=refusal)")
    text = "".join(b.text for b in resp.content if b.type == "text")
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return text, usage


def _supports_effort(model: str) -> bool:
    """The effort setting exists on the newer Claude models (Opus 4.5+, the 5.x family)."""
    model = model.replace(".", "-")
    return any(t in model for t in ("opus-4-5", "opus-4-6", "opus-4-7", "opus-4-8", "opus-5", "sonnet-5",
                                    "sonnet-4-6", "haiku-5", "fable"))


def _openai(model: str, call: Call) -> tuple[str, dict]:
    if call.images:
        raise LLMError("Image input is only wired for Claude in this project")
    client = _client("openai")
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[{"role": "system", "content": call.system + "\nReply with a JSON object."},
                  {"role": "user", "content": call.user}],
        response_format={"type": "json_object"},
        max_completion_tokens=call.max_tokens,
    )
    if model.startswith("gpt-4"):           # gpt-5 / o-series only accept the default temperature
        kwargs["temperature"] = 0
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
    return text, usage


OPENROUTER_URL = "https://openrouter.ai/api/v1"


def _openrouter(model: str, call: Call) -> tuple[str, dict]:
    """Any model on OpenRouter ("anthropic/claude-opus-5.5", "openai/gpt-4o", ...) through the OpenAI SDK."""
    client = _client("openrouter")
    content: Any = call.user
    if call.images:
        content = [{"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.standard_b64encode(Path(img).read_bytes()).decode("utf-8")}}
            for img in call.images] + [{"type": "text", "text": call.user}]
    kwargs: dict[str, Any] = dict(
        model=model, max_tokens=call.max_tokens,
        messages=[{"role": "system", "content": call.system + "\nReply with a single JSON value and nothing else."},
                  {"role": "user", "content": content}],
    )
    extra: dict[str, Any] = {"usage": {"include": True}}          # ask OpenRouter for the real cost
    if model.startswith("openai/gpt-4"):
        kwargs["temperature"] = 0
    if SETTINGS.effort and _supports_effort(model):
        extra["reasoning"] = {"effort": SETTINGS.effort}
    kwargs["extra_body"] = extra
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:                                       # auth, quota, unknown model, bad request
        raise LLMError(f"OpenRouter call failed ({model}): {type(exc).__name__}: {str(exc)[:300]}") from exc
    choice = resp.choices[0]
    text = choice.message.content or ""
    if not text.strip():
        raise LLMError(f"{model} returned no text (finish_reason={choice.finish_reason}); "
                       "the answer may have been cut off by max_tokens or filtered")
    u = resp.usage
    usage = {"input_tokens": getattr(u, "prompt_tokens", 0), "output_tokens": getattr(u, "completion_tokens", 0)}
    cost = (getattr(u, "model_extra", None) or {}).get("cost")
    if isinstance(cost, (int, float)):
        usage["usd_actual"] = float(cost)
    return text, usage


# Rough list prices, US$ per million tokens (input, output). Check current pricing before a big run;
# these are only used to estimate cost in `consistency-agent cost`.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-4-5": (3.0, 15.0), "claude-opus-4-5": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0), "claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0),
    "gpt-4o": (2.50, 10.0), "gpt-5": (1.25, 10.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    name = model.split(":", 1)[-1].replace(".", "-")
    price = next((v for k, v in PRICES.items() if k in name), None)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1e6


def _log(role: str, model: str, usage: dict, seconds: float) -> None:
    global _spent
    RUNS.mkdir(parents=True, exist_ok=True)
    cost = usage.get("usd_actual")
    if cost is None:
        cost = estimate_cost(model, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
    with _lock:
        _spent += cost or 0.0
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "model": model,
           "seconds": round(seconds, 2), "usd_est": None if cost is None else round(cost, 5), "usd_is_actual": "usd_actual" in usage, **usage}
    with _lock, open(RUNS / "usage.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
