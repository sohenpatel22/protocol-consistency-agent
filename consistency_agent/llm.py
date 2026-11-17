"""One function, complete_json(), that every step uses to call a model and get JSON back.

Claude goes through the anthropic SDK (or Bedrock), OpenAI models through the openai SDK.
Each call is logged to runs/usage.jsonl so we can report cost and time."""
from __future__ import annotations

import base64
import json
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


def _dispatch(provider: str, model: str, call: Call) -> str:
    t0 = time.time()
    if provider in ("anthropic", "bedrock"):
        text, usage = _anthropic(model, call, provider)
    elif provider == "openai":
        text, usage = _openai(model, call)
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
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=call.max_tokens,
            system=call.system,
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.BadRequestError as exc:
        raise LLMError(f"Anthropic rejected the request ({model}): {exc}") from exc
    if resp.stop_reason == "refusal":
        raise LLMError(f"{model} declined the request (stop_reason=refusal)")
    text = "".join(b.text for b in resp.content if b.type == "text")
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return text, usage


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


def _log(role: str, model: str, usage: dict, seconds: float) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "model": model,
           "seconds": round(seconds, 2), **usage}
    with _lock, open(RUNS / "usage.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
