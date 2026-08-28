"""The API path for every step that needs a model.

The design gives two paths, in order. If ANTHROPIC_API_KEY or OPENAI_API_KEY is
in the environment, use it. If neither is, a Claude Code session fills the same
payload files by hand and the pipeline carries on identically. Both paths write
the same JSON to the same place, and everything downstream validates that JSON
the same way, so which path produced a committed result changes nothing about
how much it can be trusted.

No SDK. Both APIs are one POST with a JSON body, and adding a dependency to save
forty lines would be a worse trade than the forty lines.

The committed results in this repository came from the Claude Code path, because
no key was present on the machine that built it. The API path is exercised by
`llm.py selftest`, which needs a key.

  SMAA_MODEL           overrides the model id
  SMAA_MAX_TOKENS      overrides the output cap, default 8000
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-4.1"
DEFAULT_MAX_TOKENS = 8000


class NoKey(RuntimeError):
    """Raised when neither provider has a key in the environment."""


def provider() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def available() -> bool:
    return provider() is not None


def model_id() -> str:
    override = os.environ.get("SMAA_MODEL")
    if override:
        return override
    return DEFAULT_ANTHROPIC_MODEL if provider() == "anthropic" else DEFAULT_OPENAI_MODEL


def _post(url: str, headers: dict, body: dict, retries: int = 5) -> dict:
    payload = json.dumps(body).encode()
    last: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (429, 500, 502, 503, 504, 529):
                wait = min(60, 5 * (2 ** attempt))
                print(f"    {exc.code} from the API, waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"{exc.code} from the API: {detail}") from exc
        except urllib.error.URLError as exc:
            last = exc
            time.sleep(min(60, 5 * (2 ** attempt)))
    raise RuntimeError(f"API failed after {retries} attempts: {last}")


def complete(prompt: str, system: str = "") -> str:
    """One completion. Returns the raw text the model produced."""
    which = provider()
    if which is None:
        raise NoKey("no ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment")

    max_tokens = int(os.environ.get("SMAA_MAX_TOKENS", DEFAULT_MAX_TOKENS))

    if which == "anthropic":
        body = {
            "model": model_id(),
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        data = _post(ANTHROPIC_URL, {
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }, body)
        return "".join(block.get("text", "") for block in data.get("content", []))

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    data = _post(OPENAI_URL, {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    }, {"model": model_id(), "max_tokens": max_tokens, "messages": messages})
    return data["choices"][0]["message"]["content"]


FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def complete_json(prompt: str, system: str = "") -> dict:
    """A completion that must be one JSON object.

    Models sometimes wrap JSON in a fence or add a sentence before it even when
    told not to. Strip the fence, then fall back to the outermost braces. If it
    still does not parse, that is a real failure and it is raised, not silently
    swallowed into an empty result.
    """
    text = complete(prompt, system).strip()
    fence = FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start: end + 1])
        raise


JSON_SYSTEM = (
    "You return a single valid JSON object and nothing else. No prose before it, "
    "no prose after it, and no markdown code fence around it."
)


def run_batches(in_dir, out_dir, pattern: str, suffix: str = ".txt") -> int:
    """Turn every prepared prompt file into its JSON answer file.

    Resumable: a batch whose output already exists and parses is skipped, so a
    run interrupted by a rate limit or a laptop lid picks up where it stopped.
    This is the same contract the Claude Code path honors, which is why the two
    paths are interchangeable.
    """
    from pathlib import Path
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = sorted(in_dir.glob(pattern + suffix))
    done = 0
    for i, path in enumerate(prompts, 1):
        target = out_dir / (path.name[: -len(suffix)] + ".json")
        if target.exists():
            try:
                json.loads(target.read_text(encoding="utf-8"))
                done += 1
                continue
            except json.JSONDecodeError:
                pass
        print(f"  {i}/{len(prompts)} {path.stem}", flush=True)
        result = complete_json(path.read_text(encoding="utf-8"), JSON_SYSTEM)
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        done += 1
    print(f"  {done}/{len(prompts)} batches complete")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        if not available():
            print("No ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment.")
            print("The pipeline falls back to the Claude Code path; nothing is broken.")
            return 1
        print(f"provider {provider()}, model {model_id()}")
        result = complete_json(
            'Return exactly {"ok": true, "echo": "ready"} and nothing else.',
            JSON_SYSTEM)
        print("round trip:", result)
        return 0
    print("usage: python llm.py selftest")
    return 2


if __name__ == "__main__":
    sys.exit(main())
