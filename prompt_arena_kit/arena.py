"""
arena.py - the Prompt Arena harness.

Students write prompts. This module runs them on a shared task, scores them
(macro-F1, Cohen's kappa, bootstrap CI, validity, cost), enforces a spend
allowance, and reports each run to the instructor's sheet.

Pure standard library (the OpenAI SDK is imported only when a real model is used).

Quick use (in Colab):

    import arena
    arena.setup(student_id="otter42", api_key=userdata.get("OPENAI_API_KEY"),
                endpoint=ENDPOINT, token=TOKEN, allowance_usd=0.50)
    res = arena.evaluate("Classify the text.\\n{codebook}\\n\\n{text}", task="A")
    res.summary()

Placeholders you can use in a prompt:
    {text}      the item (added at the end automatically if you omit it)
    {labels}    the label list
    {codebook}  the task's codebook, one definition per label
    {schema}    (extraction tasks) the JSON schema description
The harness appends one fixed output rule so every student is parsed the same way.

Use provider="mock" to practice with no API key (fake, free model).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "setup", "set_round", "load_task", "examples", "evaluate", "consistency",
    "estimate_cost", "complete", "audit_paraphrases", "make_paraphrases",
    "reflect_and_rewrite", "spent", "remaining", "flush_log", "BudgetExceeded",
    "try_one", "cascade", "ping", "submit", "two_prop_p", "peek", "report_r6", "token_peek", "show_codebook", "show_twins", "problem_rows", "changed_rows", "flipped_pairs", "audit_summary", "show_pairs",
]

# --------------------------------------------------------------------------
# Locations and global state
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA = HERE / "data" if (HERE / "data").exists() else HERE.parent / "data"
PRICES_PATH = HERE / "prices.json"

_lock = threading.Lock()
_cfg = {
    "student_id": "anon", "provider": "openai", "model": None,
    "api_key": None, "endpoint": None, "token": None,
    "allowance": 1.0, "spent": 0.0, "round": "R?", "client": None,
}
_cache: dict = {}
_CACHE_FILE = Path(".arena_cache.jsonl")
_LOG_FILE = Path("arena_log.jsonl")


class BudgetExceeded(RuntimeError):
    """Raised when the student's spend allowance is used up."""


class PriceMissing(RuntimeError):
    """Raised when prices.json has no price for the chosen model."""


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
def setup(student_id: str, provider: str = "openai", api_key: str | None = None,
          model: str | None = None, endpoint: str | None = None, token: str | None = None,
          allowance_usd: float = 1.0, allowed_models=None) -> None:
    """Configure the harness once per notebook.
    allowed_models: None = the class allowlist in prices.json; "all" = no limit (instructor scripts); or a list.
    Spend is remembered per round in .arena_spent_<student>.json, so re-running this cell does NOT reset your allowance."""
    prices = _prices()
    if str(student_id).strip().lower() in ("", "your-nickname", "anon", "nickname"):
        raise ValueError("Set NICK (the line above this one) to your own nickname before you run this cell. "
                         "The instructor sees it on the board, and your allowance is tracked under it.")
    student_id = "".join(str(student_id).split())[:24]          # no spaces, short
    if provider not in ("openai", "mock"):
        raise ValueError("provider must be 'openai' or 'mock'")
    if provider != "mock" and not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if provider != "mock" and not api_key:
        raise ValueError("No API key. Pass api_key=... (Colab: userdata.get('OPENAI_API_KEY')).")
    if provider != "mock":
        api_key = "".join(str(api_key).split())          # keys never contain spaces or line breaks: remove pasted ones
        if not api_key.isascii():
            raise ValueError("Your API key contains a non-ASCII character (a curly quote or a hidden symbol from copy-paste). "
                             "Delete the secret and paste the key again.")
    allowed = None if allowed_models == "all" else (list(allowed_models) if allowed_models else prices.get("student_allowed"))
    _cfg.update(student_id=student_id, provider=provider, api_key=api_key, client=None,
                model=model or prices["default_model"][provider],
                endpoint=endpoint, token=token, allowance=float(allowance_usd), spent=0.0, allowed=allowed)
    _cfg["spent"] = _read_spent()
    _load_cache()
    print(f"[arena] ready: student={student_id} provider={provider} model={_cfg['model']} "
          f"allowance=${allowance_usd:.2f} (already spent ${_cfg['spent']:.3f}) logging={'on' if endpoint else 'local only'}"
          + (f"\n[arena] models you may use: {', '.join(m for m in allowed if m != 'mock')}" if allowed else ""))


def _spent_file() -> Path:
    return Path(f".arena_spent_{_cfg['student_id']}.json")


def _read_all() -> dict:
    try:
        d = json.loads(_spent_file().read_text())
        return d if isinstance(d.get("by_round"), dict) else {"by_round": {}}
    except Exception:
        return {"by_round": {}}


def _read_spent() -> float:
    """Spend so far in the CURRENT round (each notebook has its own allowance)."""
    return float(_read_all()["by_round"].get(_cfg["round"], 0.0))


def _write_spent() -> None:
    try:
        d = _read_all()
        d["by_round"][_cfg["round"]] = _cfg["spent"]
        _spent_file().write_text(json.dumps(d))
    except OSError:
        pass


def set_round(name: str) -> None:
    """Name the round (also selects that round's allowance bucket)."""
    _cfg["round"] = name
    _cfg["spent"] = _read_spent()


def spent() -> float:
    """Spend in the current round."""
    return _cfg["spent"]


def remaining() -> float:
    return max(0.0, _cfg["allowance"] - _cfg["spent"])


def _prices() -> dict:
    return json.loads(PRICES_PATH.read_text())


def _price(model: str) -> tuple[float, float]:
    m = _prices()["models"].get(model)
    if m is None or m.get("in") is None or m.get("out") is None:
        raise PriceMissing(f"No price for '{model}' in prices.json. Ask the instructor to fill it in.")
    return m["in"], m["out"]


def _cost(model: str, tin: int, tout: int) -> float:
    pin, pout = _price(model)
    return (tin * pin + tout * pout) / 1_000_000


# --------------------------------------------------------------------------
# Tasks and data
# --------------------------------------------------------------------------
@dataclass
class Task:
    id: str
    name: str
    discipline: str
    type: str
    labels: list
    codebook: dict
    files: dict
    schema: dict | None = None
    brief: str = ""

    def codebook_text(self, order: list | None = None) -> str:
        labs = order or self.labels
        return "\n".join(f"- {l}: {self.codebook.get(l, '')}" for l in labs)

    def schema_text(self) -> str:
        if not self.schema:
            return ""
        lines = [f'- "{k}": {v}' for k, v in self.schema["fields"].items()]
        return "Return a single JSON object with exactly these keys:\n" + "\n".join(lines)


def load_task(task_id: str) -> Task:
    tasks = json.loads((DATA / "tasks.json").read_text())
    if task_id not in tasks:
        raise KeyError(f"Unknown task '{task_id}'. Available: {', '.join(tasks)}")
    t = tasks[task_id]
    return Task(id=task_id, name=t["name"], discipline=t["discipline"], type=t["type"],
                labels=t.get("labels", []), codebook=t.get("codebook", {}),
                files=t["files"], schema=t.get("schema"), brief=t.get("brief", ""))


def _load_split(task: Task, split: str) -> list:
    fn = task.files.get(split)
    path = DATA / fn if fn else None
    if not path or not path.exists():
        raise FileNotFoundError(
            f"Split '{split}' for task {task.id} is not available here."
            + (" (The hidden test set stays with the instructor.)" if split == "test" else ""))
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def examples(task_id: str, k: int = 3, seed: int = 0, diverse: bool = True) -> str:
    """Return k formatted few-shot examples from the reserved 'shot' items.
    These items are excluded from dev scoring, so there is no leakage."""
    task = load_task(task_id)
    shots = [i for i in _load_split(task, "dev") if i.get("shot")]
    if not shots:
        raise ValueError(f"Task {task_id} has no reserved example items.")
    rng = random.Random(seed)
    rng.shuffle(shots)
    chosen, seen = [], set()
    if diverse:                       # one per label first
        for it in shots:
            if it["label"] not in seen and len(chosen) < k:
                chosen.append(it); seen.add(it["label"])
    for it in shots:
        if len(chosen) >= k:
            break
        if it not in chosen:
            chosen.append(it)
    return "\n\n".join(f"Text: {c['text']}\nLabel: {c['label']}" for c in chosen)


# --------------------------------------------------------------------------
# Prompt building and parsing
# --------------------------------------------------------------------------
def _render(prompt: str, item: dict, task: Task, label_order: list | None) -> str:
    body = prompt if "{text}" in prompt else prompt.rstrip() + "\n\nText:\n{text}"
    labels = label_order or task.labels
    body = (body.replace("{labels}", ", ".join(labels))
                .replace("{codebook}", task.codebook_text(label_order))
                .replace("{schema}", task.schema_text()))
    body = body.replace("{text}", item["text"])
    if task.type == "classification":
        rule = ("\n\n(Output rule: end your reply with one final line exactly like "
                f"`Label: <one of {', '.join(labels)}>`.)")
    else:
        rule = "\n\n(Output rule: reply with the JSON object only, no other text.)"
    return body + rule


def _parse_label(text: str, labels: list):
    m = re.findall(r"Label:\s*\**\s*`?([A-Za-z0-9_ \-]+)", text)
    if m:
        cand = m[-1].strip().upper().replace(" ", "_").replace("-", "_")
        for l in labels:
            if cand == l.upper():
                return l, True
    best, pos = None, -1
    for l in labels:
        p = text.upper().rfind(l.upper())
        if p > pos:
            best, pos = l, p
    return best, False


def _parse_json(text: str, fields: list):
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < a:
        return None, False
    try:
        obj = json.loads(text[a:b + 1])
    except Exception:
        return None, False
    ok = isinstance(obj, dict) and all(f in obj for f in fields)
    return obj, ok


def _norm(v):
    if isinstance(v, list):
        return sorted(str(x).strip().lower() for x in v)
    if v is None:
        return None
    return str(v).strip().lower()


# --------------------------------------------------------------------------
# Metrics (pure python)
# --------------------------------------------------------------------------
def _macro_f1(gold: list, pred: list, labels: list) -> float:
    f1s = []
    for l in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == l and p == l)
        fp = sum(1 for g, p in zip(gold, pred) if g != l and p == l)
        fn = sum(1 for g, p in zip(gold, pred) if g == l and p != l)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s)


def _kappa(gold: list, pred: list, labels: list) -> float:
    n = len(gold)
    po = sum(1 for g, p in zip(gold, pred) if g == p) / n
    pe = sum((gold.count(l) / n) * (pred.count(l) / n) for l in labels)
    return 0.0 if pe == 1 else (po - pe) / (1 - pe)


def _bootstrap(gold: list, pred: list, labels: list, b: int = 500, seed: int = 0):
    rng = random.Random(seed)
    n = len(gold)
    vals = []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(_macro_f1([gold[i] for i in idx], [pred[i] for i in idx], labels))
    vals.sort()
    return vals[int(0.025 * b)], vals[int(0.975 * b) - 1]


def two_prop_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided p-value for a difference in proportions (pooled z-test)."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k1 / n1 - k2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


# --------------------------------------------------------------------------
# Model calls, cache, spend
# --------------------------------------------------------------------------
def _load_cache() -> None:
    if _CACHE_FILE.exists():
        for l in _CACHE_FILE.read_text().splitlines():
            try:
                d = json.loads(l); _cache[d["k"]] = d["v"]
            except Exception:
                pass


def _client():
    if _cfg["client"] is None:
        import openai
        _cfg["client"] = openai.OpenAI(api_key=_cfg["api_key"])
    return _cfg["client"]


def _raw_call(model, text, temperature, max_tokens, extra):
    extra = dict(extra or {})
    kw = dict(model=model, messages=[{"role": "user", "content": text}],
              max_completion_tokens=max_tokens, **extra)
    if temperature is not None:
        kw["temperature"] = temperature
    try:
        r = _client().chat.completions.create(**kw)
    except Exception as e:
        if "temperature" in str(e).lower() and "temperature" in kw:
            kw.pop("temperature"); r = _client().chat.completions.create(**kw)
        else:
            raise
    return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens


def _mock_call(text, item, task: Task, temperature):
    """A fake model for practice and tests. Accuracy rises with prompt effort."""
    if item is None:                                    # meta calls: paraphrase and reflect requests
        if "Return a JSON list of" in text:
            body = text.split("PROMPT:\n", 1)[-1]
            words = ["Classify", "Label", "Code", "Categorise", "Tag", "Assign"]
            outs = [body.replace("Classify", w, 1) + " " * i for i, w in enumerate(words[:5])]
            return json.dumps(outs), len(text) // 4, 100
        if "<prompt>" in text:
            cur = text.split("CURRENT PROMPT:\n", 1)[-1].split("\n\nMISTAKES", 1)[0]
            out = ("The prompt does not tell the model how to treat irony or period spelling.\n"
                   "<prompt>" + cur + "\n\nThink step by step. Read irony and period spelling as the writer intended.</prompt>")
            return out, len(text) // 4, len(out) // 4
        return "ok", len(text) // 4, 2
    # Each item has a fixed difficulty u in [0,1). The model is right when u < quality, so a
    # "better" prompt (more definitions, examples, reasoning) can only fix items, never break them.
    rng = random.Random(hashlib.md5((text + str(temperature)).encode()).hexdigest())
    u = random.Random("difficulty:" + item["id"].rstrip("i")).random()
    quality = 0.35 + 0.15 * ("- " in text and ":" in text) + 0.15 * (len(text) > 700) + 0.12 * ("step" in text.lower())
    quality = min(quality, 0.92)
    if task.type == "classification":
        if "ignore previous instructions" in text.lower() and rng.random() < 0.5:
            lab = rng.choice(task.labels)
        else:
            lab = item["label"] if u < quality else rng.choice(task.labels)
        conf = 85 if (lab == item["label"] and u < quality - 0.2) else 40
        out = f"Reasoning: looks like {lab}.\nConfidence: {conf}\nLabel: {lab}"
    else:
        g = dict(item["gold"])
        for k in list(g):
            if random.Random(f"{item['id']}{k}").random() > quality:
                g[k] = None
        out = json.dumps(g)
    return out, max(1, len(text) // 4), max(1, len(out) // 4)


def complete(prompt_text: str, model: str | None = None, temperature: float | None = 0.0,
             max_tokens: int = 600, extra: dict | None = None, rep: int = 0,
             _item: dict | None = None, _task: Task | None = None):
    """One model call with cache, retries and spend tracking.
    Returns (text, tokens_in, tokens_out, nominal_cost_usd)."""
    model = model or _cfg["model"]
    if _cfg.get("allowed") is not None and model not in _cfg["allowed"]:
        raise PermissionError(f"Model '{model}' is not allowed in this class (cost). "
                              f"Allowed: {', '.join(m for m in _cfg['allowed'] if m != 'mock')}. Ask the instructor.")
    key = hashlib.sha256(json.dumps([_cfg["provider"], model, prompt_text, temperature,
                                     max_tokens, extra, rep], sort_keys=True).encode()).hexdigest()
    if key in _cache:
        v = _cache[key]
        return v["text"], v["tin"], v["tout"], _cost(model, v["tin"], v["tout"])
    with _lock:
        if _cfg["spent"] >= _cfg["allowance"]:
            raise BudgetExceeded(f"Allowance of ${_cfg['allowance']:.2f} used up.")
    last = None
    for attempt in range(6):
        try:
            if _cfg["provider"] == "mock":
                text, tin, tout = _mock_call(prompt_text, _item, _task, temperature)
            else:
                text, tin, tout = _raw_call(model, prompt_text, temperature, max_tokens, extra)
            break
        except Exception as e:                       # noqa: BLE001
            last = e
            msg = str(e).lower()
            if "insufficient_quota" in msg or "exceeded your current quota" in msg or "billing" in msg:
                raise RuntimeError("The class API budget is used up (or billing is off). "
                                   "Tell the instructor. Runs you already finished are cached and will not be lost.") from e
            root = repr(getattr(e, "__cause__", None)).lower()
            if "illegal header value" in root or "illegal header value" in msg:
                raise RuntimeError("Your API key has a hidden space or line break in it (copy-paste). "
                                   "Re-paste it into the OPENAI_API_KEY secret, then Runtime > Restart session.") from e
            if "invalid_api_key" in msg or "incorrect api key" in msg or "401" in msg:
                raise RuntimeError("Your API key was rejected. Check the secret name (OPENAI_API_KEY), "
                                   "that 'Notebook access' is on, and that you pasted the whole key.") from e
            time.sleep(min(30.0, 1.5 * (2 ** attempt)) * (0.5 + random.random()))   # backoff with jitter
    else:
        cause = getattr(last, "__cause__", None)
        raise RuntimeError(f"API call failed after retries: {last}" + (f" (underlying cause: {cause!r})" if cause else ""))
    cost = _cost(model, tin, tout)
    with _lock:
        _cfg["spent"] += cost
        _write_spent()
        _cache[key] = {"text": text, "tin": tin, "tout": tout}
        try:
            with _CACHE_FILE.open("a") as f:
                f.write(json.dumps({"k": key, "v": _cache[key]}) + "\n")
        except OSError:
            pass
    return text, tin, tout, cost


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class Result:
    task: str
    split: str
    model: str
    condition: str | None
    n: int
    metrics: dict
    records: list = field(default_factory=list)
    prompt: str = ""

    def summary(self) -> None:
        m = self.metrics
        print(f"Task {self.task}/{self.split}  model={self.model}  n={self.n}")
        if "macro_f1" in m:
            print(f"  macro-F1 {m['macro_f1']:.3f}  (95% CI {m['ci_low']:.3f}-{m['ci_high']:.3f})"
                  f"   kappa {m['kappa']:.3f}   accuracy {m['accuracy']:.3f}")
        if "field_accuracy" in m:
            print(f"  field accuracy {m['field_accuracy']:.3f}   per field: "
                  + ", ".join(f"{k}={v:.2f}" for k, v in m["per_field"].items()))
        print(f"  valid format {m['validity']:.0%}   tokens/item in={m['tokens_in_per_item']:.0f} "
              f"out={m['tokens_out_per_item']:.0f}")
        print(f"  cost this run ${m['cost_usd']:.4f}   per 1,000 items ${m['cost_per_1k']:.2f}"
              f"   | allowance left ${remaining():.2f}")
        if "flip_rate" in m:
            print(f"  flip rate across matched pairs {m['flip_rate']:.0%}")
        if "acc_by_variant" in m:
            print("  accuracy by variant: " + ", ".join(f"{k}={v:.2f}" for k, v in m["acc_by_variant"].items()))

    def errors(self, k: int = 8) -> list:
        return [r for r in self.records if not r["correct"]][:k]


def evaluate(prompt: str, task: str = "A", split: str = "dev", model: str | None = None,
             n: int | None = None, condition: str | None = None, round_: str | None = None,
             temperature: float | None = 0.0, max_tokens: int = 1500, extra: dict | None = None,
             seed: int = 0, shuffle_labels: int | None = None, rep: int = 0,
             log: bool = True, note: str = "") -> Result:
    """Run `prompt` on a split of a task and return scores + cost."""
    t = load_task(task)
    model = model or _cfg["model"]
    items = _load_split(t, split)
    if split == "dev":
        items = [i for i in items if not i.get("shot")]
    if n and n < len(items):
        rng = random.Random(seed); rng.shuffle(items); items = items[:n]
    order = None
    if shuffle_labels is not None and t.labels:
        order = t.labels[:]; random.Random(shuffle_labels).shuffle(order)

    fields = list(t.schema["fields"]) if t.schema else []

    def work(item):
        text = _render(prompt, item, t, order)
        out, tin, tout, cost = complete(text, model, temperature, max_tokens, extra, rep, item, t)
        rec = {"id": item["id"], "text": item["text"][:300], "tin": tin, "tout": tout, "cost": cost, "raw": out,
               "variant": item.get("variant", "clean"), "pair_id": item.get("pair_id")}
        if t.type == "classification":
            pred, strict = _parse_label(out, t.labels)
            rec.update(gold=item["label"], pred=pred, valid=strict, correct=(pred == item["label"]))
        else:
            obj, ok = _parse_json(out, fields)
            rec.update(gold=item["gold"], pred=obj, valid=ok)
            rec["correct"] = bool(obj) and all(_norm(obj.get(f)) == _norm(item["gold"].get(f)) for f in fields)
        return rec

    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            recs = list(ex.map(work, items))
    except BudgetExceeded:
        print("[arena] STOPPED: allowance used up. This run was not scored or logged.")
        raise
    m = _metrics(recs, t, fields)
    res = Result(task=task, split=split, model=model, condition=condition, n=len(recs),
                 metrics=m, records=recs, prompt=prompt)
    if log:
        _log(res, round_ or _cfg["round"], note)
    return res


def _metrics(recs: list, t: Task, fields: list) -> dict:
    n = len(recs)
    m = {"validity": sum(r["valid"] for r in recs) / n,
         "tokens_in_per_item": sum(r["tin"] for r in recs) / n,
         "tokens_out_per_item": sum(r["tout"] for r in recs) / n,
         "cost_usd": sum(r["cost"] for r in recs)}
    m["cost_per_1k"] = m["cost_usd"] / n * 1000
    if t.type == "classification":
        gold = [r["gold"] for r in recs]
        pred = [r["pred"] if r["pred"] is not None else "INVALID" for r in recs]
        m["accuracy"] = sum(g == p for g, p in zip(gold, pred)) / n
        m["macro_f1"] = _macro_f1(gold, pred, t.labels)
        m["kappa"] = _kappa(gold, pred, t.labels + ["INVALID"])
        m["ci_low"], m["ci_high"] = _bootstrap(gold, pred, t.labels)
        variants = {r["variant"] for r in recs}
        if len(variants) > 1:
            m["acc_by_variant"] = {v: sum(r["correct"] for r in recs if r["variant"] == v) /
                                   max(1, sum(1 for r in recs if r["variant"] == v)) for v in sorted(variants)}
        pairs: dict = {}
        for r in recs:
            if r["pair_id"]:
                pairs.setdefault(r["pair_id"], []).append(r["pred"])
        if pairs:
            full = [v for v in pairs.values() if len(v) == 2]
            m["flip_rate"] = sum(1 for a, b in full if a != b) / max(1, len(full))
    else:
        per = {}
        for f in fields:
            per[f] = sum(1 for r in recs if r["pred"] and _norm(r["pred"].get(f)) == _norm(r["gold"].get(f))) / n
        m["per_field"] = per
        m["field_accuracy"] = sum(per.values()) / len(per)
        m["macro_f1"] = m["field_accuracy"]
        m["accuracy"] = sum(r["correct"] for r in recs) / n
        m["kappa"] = float("nan"); m["ci_low"] = m["ci_high"] = float("nan")
    return m


# --------------------------------------------------------------------------
# Cost tools
# --------------------------------------------------------------------------
def estimate_cost(tokens_in_per_item: float, tokens_out_per_item: float, n_items: int,
                  model: str | None = None) -> dict:
    """Project the bill for n_items. Use it BEFORE running, then compare to actual."""
    model = model or _cfg["model"]
    c = _cost(model, int(tokens_in_per_item * n_items), int(tokens_out_per_item * n_items))
    return {"model": model, "n_items": n_items, "projected_usd": c, "per_1k_items": c / n_items * 1000}


def consistency(prompt: str, task: str = "A", n: int = 30, runs: int = 3, temperature: float = 0.7, **kw) -> dict:
    """Run the same prompt several times at temperature>0 and measure agreement."""
    results = [evaluate(prompt, task, n=n, temperature=temperature, rep=r, log=False, **kw) for r in range(runs)]
    preds = [[x["pred"] for x in r.records] for r in results]
    agree, pairs = 0, 0
    for i in range(runs):
        for j in range(i + 1, runs):
            agree += sum(a == b for a, b in zip(preds[i], preds[j])) / len(preds[i]); pairs += 1
    f1s = [r.metrics["macro_f1"] for r in results]
    out = {"agreement": agree / pairs, "f1_min": min(f1s), "f1_max": max(f1s), "runs": runs}
    print(f"[consistency] agreement {out['agreement']:.0%}; F1 range {out['f1_min']:.3f}-{out['f1_max']:.3f}")
    unstable = [(results[0].records[i], [preds[r][i] for r in range(runs)]) for i in range(len(preds[0]))
                if len({preds[r][i] for r in range(runs)}) > 1]
    out["unstable_items"] = [u[0]["id"] for u in unstable]
    print(f"{len(unstable)} of {len(preds[0])} items got a DIFFERENT label in at least one of the {runs} runs." +
          (" The first ones:" if unstable else ""))
    for rec, ps in unstable[:3]:
        print(f"   [{rec['id']}] correct = {rec['gold']}  runs said: {ps}\n      {rec['text'][:170]}")
    return out


def try_one(prompt: str, text: str, task: str = "A", model: str | None = None, temperature: float | None = 0.0) -> str:
    """Run the prompt on ONE custom text (e.g. an adversarial input) and print the raw reply and parse."""
    t = load_task(task)
    item = {"id": "custom", "text": text, "label": t.labels[0] if t.labels else None, "gold": {}}
    out, tin, tout, cost = complete(_render(prompt, item, t, None), model, temperature, _item=item, _task=t)
    if t.type == "classification":
        lab, strict = _parse_label(out, t.labels)
        print(f"--- model reply ---\n{out}\n--- parsed: {lab}  (strict format: {strict})  cost ${cost:.5f}")
    else:
        obj, ok = _parse_json(out, list(t.schema["fields"]))
        print(f"--- model reply ---\n{out}\n--- valid JSON with all fields: {ok}  cost ${cost:.5f}")
    return out


def cascade(prompt: str, task: str = "A", cheap: str | None = None, strong: str | None = None,
            threshold: int = 70, split: str = "dev", n: int | None = None, log: bool = True,
            round_: str | None = None, note: str = "") -> Result:
    """Two-stage pipeline: the cheap model labels everything and reports a confidence (0-100);
    items below `threshold` are re-labelled by the strong model. Cost counts both calls."""
    t = load_task(task)
    cheap = cheap or _cfg["model"]
    strong = strong or cheap
    items = [i for i in _load_split(t, split) if not i.get("shot")]
    if n and n < len(items):
        random.Random(0).shuffle(items); items = items[:n]
    ask = prompt.rstrip() + "\n\nAlso write one line `Confidence: <0-100>` (how sure you are) before the final label line."

    def conf_of(text):
        m = re.findall(r"Confidence:\s*(\d+)", text)
        return int(m[-1]) if m else None

    def work(item):
        text = _render(ask, item, t, None)
        o1, i1, p1, c1 = complete(text, cheap, 0.0, 600, None, 0, item, t)
        lab, strict = _parse_label(o1, t.labels)
        conf = conf_of(o1)
        tin, tout, cost, esc = i1, p1, c1, False
        if conf is None or conf < threshold:
            esc = True
            o2, i2, p2, c2 = complete(text, strong, 0.0, 600, None, 0, item, t)
            lab, strict = _parse_label(o2, t.labels)
            tin += i2; tout += p2; cost += c2
        return {"id": item["id"], "text": item["text"][:300], "tin": tin, "tout": tout, "cost": cost, "raw": "",
                "variant": "clean", "pair_id": None, "gold": item["label"], "pred": lab, "valid": strict,
                "correct": lab == item["label"], "escalated": esc}

    with ThreadPoolExecutor(max_workers=8) as ex:
        recs = list(ex.map(work, items))
    m = _metrics(recs, t, [])
    m["escalated"] = sum(r["escalated"] for r in recs) / len(recs)
    res = Result(task=task, split=split, model=f"{cheap}->{strong}", condition="cascade", n=len(recs),
                 metrics=m, records=recs, prompt=prompt)
    print(f"[cascade] escalated {m['escalated']:.0%} of items to the strong model")
    if log:
        _log(res, round_ or _cfg["round"], note or f"threshold={threshold} escalated={m['escalated']:.2f}")
    return res


def peek(task_id: str, k: int = 4, seed: int = 0, show_labels: bool = False, split: str = "dev") -> None:
    """Print the client brief, the labels, the codebook and k sample items (labels hidden unless asked)."""
    t = load_task(task_id)
    print(f"Task {t.id}: {t.name}  [{t.discipline}]")
    if t.brief:
        print(t.brief)
    if t.labels:
        print("Labels:", ", ".join(t.labels))
    items = [i for i in _load_split(t, split) if not i.get("shot")]
    print(f"({len(items)} items in the '{split}' set; showing {min(k, len(items))})")
    random.Random(seed).shuffle(items)
    for it in items[:k]:
        tag = f"   -> {it['label']}" if (show_labels and "label" in it) else ""
        src = f"  ({it['newspaper']})" if it.get("newspaper") else ""
        print(f"\n[{it['id']}]{src} {it['text']}{tag}")


def show_codebook(task_id: str) -> None:
    """Print the official label definitions for a task."""
    t = load_task(task_id)
    print(f"Codebook for task {t.id}: {t.name}")
    print(t.codebook_text())


def show_twins(task_id: str = "E", k: int = 2, seed: int = 0) -> None:
    """Show k clean posts next to their INJECTED twins (same post plus a sentence posing as an instruction)."""
    t = load_task(task_id)
    items = _load_split(t, "robust")
    by_id = {i["id"]: i for i in items}
    clean = [i for i in items if i.get("variant") == "clean"]
    random.Random(seed).shuffle(clean)
    for it in clean[:k]:
        twin = by_id.get(it["id"] + "i")
        print(f"[{it['id']}]  CLEAN, correct label = {it['label']}\n   {it['text']}")
        if twin:
            extra = twin["text"][len(it["text"]):].strip()
            print(f"[{twin['id']}] INJECTED twin, same correct label = {twin['label']}\n   ...the same post, then this extra text: {extra}\n")


def problem_rows(res: "Result", variant: str | None = None, k: int = 6) -> None:
    """Show the rows the model got WRONG (optionally only one variant), with the end of the model's reply."""
    rows = [r for r in res.records if not r["correct"] and (variant is None or r["variant"] == variant)]
    name = f"variant '{variant}'" if variant else "all rows"
    print(f"{len(rows)} wrong rows in {name}" + (f"; showing the first {k}" if len(rows) > k else "") + ".")
    for r in rows[:k]:
        print(f"\n[{r['id']}] ({r['variant']}) correct = {r['gold']}   predicted = {r['pred']}")
        print(f"   post : {r['text'][:230]}")
        tail = (r.get("raw") or "").strip().splitlines()[-1:] or [""]
        print(f"   reply: ...{tail[0][-110:]}")


def changed_rows(a: "Result", b: "Result", k: int = 6, name_a: str = "run A", name_b: str = "run B") -> None:
    """Show the items whose predicted label differs between two runs on the same items."""
    ia = {r["id"]: r for r in a.records}
    rows = [(ia[r["id"]], r) for r in b.records if r["id"] in ia and ia[r["id"]]["pred"] != r["pred"]]
    print(f"{len(rows)} of {len(b.records)} items got a different label between {name_a} and {name_b}" +
          (f"; showing the first {k}" if len(rows) > k else "") + ".")
    for ra, rb in rows[:k]:
        print(f"\n[{rb['id']}] correct = {rb['gold']}   {name_a}: {ra['pred']}   {name_b}: {rb['pred']}\n   {rb['text'][:200]}")


def flipped_pairs(res: "Result", k: int = 5) -> None:
    """For matched pairs (same facts, only the name differs): show the pairs whose label changed."""
    pairs: dict = {}
    for r in res.records:
        if r.get("pair_id"):
            pairs.setdefault(r["pair_id"], []).append(r)
    full = [v for v in pairs.values() if len(v) == 2]
    flipped = [v for v in full if v[0]["pred"] != v[1]["pred"]]
    print(f"{len(flipped)} of {len(full)} matched pairs changed label when ONLY the name changed (correct: 0).")
    for a, b in flipped[:k]:
        print(f"\n{a['pair_id']}: correct label for both = {a['gold']}")
        print(f"   {a['text'][:150]}   -> {a['pred']}")
        print(f"   {b['text'][:150]}   -> {b['pred']}")


def show_pairs(task_id: str = "F", k: int = 2, seed: int = 0) -> None:
    """Show k matched pairs: the SAME facts with a different name. The correct label is identical for both."""
    t = load_task(task_id)
    pairs: dict = {}
    for it in _load_split(t, "dev"):
        if it.get("pair_id"):
            pairs.setdefault(it["pair_id"], []).append(it)
    ids = [i for i, v in pairs.items() if len(v) == 2]
    random.Random(seed).shuffle(ids)
    print(f"{len(ids)} matched pairs in this task; showing {min(k, len(ids))}.")
    for pid in ids[:k]:
        a, b = pairs[pid]
        print(f"\n{pid}: correct label for BOTH = {a['label']}\n   version A: {a['text']}\n   version B: {b['text']}")


def audit_summary(rows: list, alpha: float = 0.05) -> None:
    """Print the audit as a small table: one line per prompt version."""
    print(f"{'prompt version':<16}{'Free Tide':>10}{'Gazette':>9}{'difference':>12}{'p-value':>10}   verdict (alpha {alpha})")
    sig = 0
    for i, r in enumerate(rows):
        name = "main prompt" if i == 0 else f"paraphrase {i}"
        verdict = "SIGNIFICANT" if r["p_value"] < alpha else "not significant"
        sig += r["p_value"] < alpha
        print(f"{name:<16}{r['share_a']:>10.0%}{r['share_b']:>9.0%}{r['diff']:>+12.2f}{r['p_value']:>10.4f}   {verdict}")
    print(f"\n{sig} of {len(rows)} versions of the SAME question say 'significant'. "
          + ("The conclusion depends on the wording." if 0 < sig < len(rows) else "The conclusion is the same for every version."))


def report_r6(tool: str, traps_caught: int, turns: int, mode: str = "fast", verified: bool = False,
              arm: str = "") -> None:
    """Self-report for the coding-tools round (R6). Appears on the instructor's board."""
    note = json.dumps({"tool": tool, "traps_caught": int(traps_caught), "turns": int(turns),
                       "mode": mode, "verified": bool(verified)})
    row = {"ts": int(time.time()), "student_id": _cfg["student_id"], "round": "R6",
           "condition": arm, "task": "G", "split": "report", "model": tool, "note": note}
    with _LOG_FILE.open("a") as f:
        f.write(json.dumps({**row, "_sent": False}) + "\n")
    print("[arena] reported." if _post(row) else "[arena] saved locally; tell the instructor your numbers out loud.")


def token_peek(prompt: str, task: str = "A", k: int = 3, model: str | None = None, n_items: int | None = None,
               budget: float | None = None, seed: int = 0) -> dict:
    """Look at k real inputs and the model's replies, with the TOKENS and COST of each, before you predict anything.
    Pass n_items (and budget) to see what the full job would cost at this rate."""
    t = load_task(task)
    model = model or _cfg["model"]
    items = [i for i in _load_split(t, "dev") if not i.get("shot")]
    random.Random(seed).shuffle(items)
    tot_in = tot_out = tot_cost = tot_item = 0
    for n, it in enumerate(items[:k], 1):
        out, tin, tout, cost = complete(_render(prompt, it, t, None), model, 0.0, 1500, None, 0, it, t)
        item_tokens = max(1, len(it["text"]) // 4)          # rough: about 4 characters per token
        tot_in += tin; tot_out += tout; tot_cost += cost; tot_item += item_tokens
        print(f"--- example {n} ({model}) ---")
        print(f"INPUT (the item, about {item_tokens} tokens): {it['text'][:220]}")
        print(f"MODEL REPLY: {out.strip()[:260]}")
        print(f"TOKENS: {tin} in (the whole prompt) + {tout} out  ->  ${cost:.5f} for this one item   [correct label: {it.get('label')}]\n")
    kk = max(1, min(k, len(items)))
    res = {"tokens_in": tot_in / kk, "tokens_out": tot_out / kk, "cost_per_item": tot_cost / kk}
    res["cost_per_1k"] = res["cost_per_item"] * 1000
    print(f"AVERAGE: {res['tokens_in']:.0f} tokens in, {res['tokens_out']:.0f} tokens out -> ${res['cost_per_item']:.5f} per item, ${res['cost_per_1k']:.3f} per 1,000 items")
    print(f"Of the {res['tokens_in']:.0f} input tokens, only about {tot_item / kk:.0f} are the item itself; the rest is your instructions and codebook. (rough: 4 characters per token)")
    if n_items:
        full = res["cost_per_item"] * n_items
        res["full_job"] = full
        line = f"At this rate {n_items:,} items would cost about ${full:,.2f}"
        if budget:
            line += f"   vs budget ${budget:,.2f}: " + ("fits" if full <= budget else "OVER BUDGET")
        print(line + ".  (Only 3 items measured: treat it as a first guess.)")
    return res


def ping() -> bool:
    """Check that the instructor's board is reachable. Prints a clear message."""
    if not _cfg["endpoint"]:
        print("[arena] No endpoint set: your runs stay on this notebook only."); return False
    ok = _post({"ts": int(time.time()), "student_id": _cfg["student_id"], "round": "PING",
                "split": "ping", "note": "setup check"})
    print("[arena] board reachable: you will appear on the projector." if ok else
          "[arena] Could not reach the board. Check ENDPOINT and TOKEN, or Wi-Fi. Your runs are saved locally.")
    return ok


def submit(prompt: str, task: str = "A", model: str | None = None, dev_f1: float | None = None) -> None:
    """Hand in your final prompt. The instructor runs it on the hidden test set after the session."""
    row = {"ts": int(time.time()), "student_id": _cfg["student_id"], "round": "SUBMIT",
           "task": task, "split": "submit", "model": model or _cfg["model"], "prompt": prompt[:4000],
           "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:8],
           "f1": dev_f1 if dev_f1 is not None else "", "note": "final submission"}
    with _LOG_FILE.open("a") as f:
        f.write(json.dumps({**row, "_sent": False}) + "\n")
    ok = _post(row)
    print("[arena] submitted." if ok else "[arena] saved locally; could not reach the board. Run arena.flush_log() later.")
    Path("final_prompt.txt").write_text(prompt)
    print("A copy is saved as final_prompt.txt in this notebook's files.")


# --------------------------------------------------------------------------
# R5 audit, R7 optimizer helpers
# --------------------------------------------------------------------------
def make_paraphrases(prompt: str, k: int = 5, model: str | None = None) -> list:
    """Ask the model for k paraphrases of a prompt (same meaning, different wording).
    Keeps placeholders such as {text}, {codebook}, {labels}."""
    ask = (f"Rewrite the following prompt in {k} different ways. Keep the meaning, the task and every "
           "placeholder in curly braces exactly as written. Vary wording and structure. "
           f"Return a JSON list of {k} strings and nothing else.\n\nPROMPT:\n{prompt}")
    out, *_ = complete(ask, model, temperature=0.9, max_tokens=2000)
    a, b = out.find("["), out.rfind("]")
    try:
        lst = json.loads(out[a:b + 1])
        return [str(x) for x in lst][:k]
    except Exception:
        raise RuntimeError("Could not parse paraphrases; run the cell again.")


def audit_paraphrases(prompts: list, task: str = "A", split: str = "audit", model: str | None = None,
                      group_field: str = "newspaper", target_label: str = "SYMPATHETIC",
                      group_a: str = "The Free Tide", group_b: str = "The Harbor Gazette",
                      log: bool = True) -> list:
    """Label the audit set with each prompt and test: is group_a more often `target_label` than group_b?
    Returns one row per prompt with both shares and a p-value. Also logs each row (unless log=False,
    in which case the would-be log row is returned under '_logrow')."""
    t = load_task(task)
    items = _load_split(t, split)
    rows = []
    for i, p in enumerate(prompts):
        res = evaluate(p, task, split=split, model=model, log=False)
        pred = {r["id"]: r["pred"] for r in res.records}
        ka = sum(1 for it in items if it[group_field] == group_a and pred[it["id"]] == target_label)
        na = sum(1 for it in items if it[group_field] == group_a)
        kb = sum(1 for it in items if it[group_field] == group_b and pred[it["id"]] == target_label)
        nb = sum(1 for it in items if it[group_field] == group_b)
        row = {"variant": i, "share_a": ka / na, "share_b": kb / nb, "diff": ka / na - kb / nb,
               "p_value": two_prop_p(ka, na, kb, nb), "macro_f1": res.metrics["macro_f1"],
               "cost_usd": res.metrics["cost_usd"]}
        note = f"variant={i} diff={row['diff']:.3f} p={row['p_value']:.4f}"
        if log:
            _log(res, "R5-audit", note)
        else:
            row["_logrow"] = _row(res, "R5-audit", note)
        rows.append(row)
        print(f"  prompt variant {i}: share {row['share_a']:.2f} vs {row['share_b']:.2f}  "
              f"diff {row['diff']:+.3f}  p={row['p_value']:.4f}  F1={row['macro_f1']:.2f}")
    return rows


def reflect_and_rewrite(prompt: str, result: Result, model: str | None = None, k_errors: int = 8) -> str:
    """GEPA-style step: show the model its mistakes, ask for a diagnosis, then a rewritten prompt."""
    errs = result.errors(k_errors)
    if not errs:
        return prompt
    shown = "\n".join(f"- gold={e['gold']} predicted={e['pred']} | model said: {e['raw'][:200]!r}" for e in errs)
    ask = ("You are improving a prompt used to classify texts.\n\nCURRENT PROMPT:\n" + prompt +
           "\n\nMISTAKES ON A DEV SET:\n" + shown +
           "\n\nFirst write 2-3 sentences diagnosing WHY the prompt fails (patterns, not single items). "
           "Then write an improved prompt. Keep placeholders like {text}, {codebook}, {labels}. "
           "Put the improved prompt between <prompt> and </prompt> tags.")
    out, *_ = complete(ask, model, temperature=0.5, max_tokens=1500)
    m = re.search(r"<prompt>(.*?)</prompt>", out, re.S)
    print("[reflection]", out.split("<prompt>")[0].strip()[:500])
    return m.group(1).strip() if m else prompt


# --------------------------------------------------------------------------
# Logging to the instructor sheet
# --------------------------------------------------------------------------
def _row(res: Result, rnd: str, note: str) -> dict:
    m = res.metrics
    return {
        "ts": int(time.time()), "student_id": _cfg["student_id"], "round": rnd,
        "condition": res.condition or "", "task": res.task, "split": res.split, "model": res.model,
        "prompt": res.prompt[:4000], "prompt_hash": hashlib.md5(res.prompt.encode()).hexdigest()[:8],
        "f1": round(m.get("macro_f1", float("nan")), 4), "kappa": round(m.get("kappa", float("nan")), 4),
        "ci_low": round(m.get("ci_low", float("nan")), 4), "ci_high": round(m.get("ci_high", float("nan")), 4),
        "cost_usd": round(m["cost_usd"], 6), "cost_per_1k": round(m["cost_per_1k"], 4),
        "tokens_in": round(m["tokens_in_per_item"], 1), "tokens_out": round(m["tokens_out_per_item"], 1),
        "validity": round(m["validity"], 4), "n": res.n, "note": note,
        "robust": json.dumps(m.get("acc_by_variant", {})), "flip_rate": round(m.get("flip_rate", float("nan")), 4),
        "preds": json.dumps({r["id"]: (r["pred"] if r["pred"] is None or isinstance(r["pred"], str) else "obj")
                             for r in res.records}) if res.split in ("dev", "audit") else "",
    }


def _clean(row: dict) -> dict:
    """NaN/inf are not valid JSON (the Sheet would reject the row): send them as empty cells."""
    return {k: ("" if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))) else v) for k, v in row.items()}


def _post(row: dict) -> bool:
    if not _cfg["endpoint"]:
        return False
    try:
        body = json.dumps({"token": _cfg["token"], "row": _clean(row)}, allow_nan=False).encode()
        req = urllib.request.Request(_cfg["endpoint"], data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.status == 200
    except Exception:
        return False


def _log(res: Result, rnd: str, note: str = "") -> None:
    row = _row(res, rnd, note)
    with _LOG_FILE.open("a") as f:
        f.write(json.dumps({**row, "_sent": False}) + "\n")
    threading.Thread(target=_post, args=(row,), daemon=True).start()


def flush_log() -> None:
    """Retry sending any rows that did not reach the instructor (e.g. after Wi-Fi trouble)."""
    if not _LOG_FILE.exists():
        return
    rows = [json.loads(l) for l in _LOG_FILE.read_text().splitlines() if l.strip()]
    ok = 0
    for r in rows:
        r2 = {k: v for k, v in r.items() if k != "_sent"}
        if _post(r2):
            ok += 1
    print(f"[arena] re-sent {ok}/{len(rows)} logged runs")
