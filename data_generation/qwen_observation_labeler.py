"""Generate Qwen3-12B observation labels for MIMIC discharge notes."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from data_generation.paths import MIMIC_DATA_DIR, OUTPUT_DIR, REPO_ROOT, ensure_repo_on_path

ensure_repo_on_path()

OBSERVATION_LABEL_MODEL = "qwen3-12b"
DEFAULT_MODEL = "Qwen/Qwen3-12B-Instruct"
PROMPT_PATH = REPO_ROOT / "prompts" / "observation_status.txt"
CONFIG_PATH = REPO_ROOT / "configs" / "observation_labeling.yaml"
LABELS_PATH = MIMIC_DATA_DIR / "mimic_qwen3_labels.jsonl"
STRUCT_PATH = MIMIC_DATA_DIR / "mimic_struct_qwen3.jsonl"
SOURCE_STRUCT = MIMIC_DATA_DIR / "mimic_struct.jsonl"

QWEN3_TO_MENTION = {
    "observed_positive": "observed_pos",
    "observed_negative": "observed_neg",
    "unmentioned": "unmentioned",
    "historical_only": "observed_historical",
    "admission_only": "observed_historical",
    "uncertain": "observed_uncertain",
    "temporal_conflict": "observed_mixed",
}

ALLOWED_STATUSES = set(QWEN3_TO_MENTION)


def load_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    from data_generation.silentehr import OBSERVATION_PROMPT  # noqa: WPS433

    return OBSERVATION_PROMPT.strip()


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    return {}


def _span_ok(span: str, text: str) -> bool:
    if not span:
        return True
    if span in text:
        return True
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    return norm(span) in norm(text)


def _parse_labels(raw: str, concepts: list[str], text: str) -> list[dict]:
    obj = json.loads(raw)
    labels = obj.get("labels", obj if isinstance(obj, list) else [])
    out = []
    seen = set()
    for lab in labels:
        concept = lab.get("concept")
        status = lab.get("observation_status")
        if concept not in concepts:
            continue
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"invalid status {status!r} for {concept}")
        span = (lab.get("evidence_span") or "").strip()
        if span and not _span_ok(span, text):
            span = ""
            lab = dict(lab)
            lab["evidence_span"] = ""
            lab["span_unverified"] = True
        seen.add(concept)
        out.append(lab)
    missing = set(concepts) - seen
    if missing:
        raise ValueError(f"missing concepts in response: {sorted(missing)}")
    return out


class Qwen3ObservationLabeler:
    """Local Qwen3-12B-Instruct labeler via Hugging Face transformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 768,
        temperature: float = 0.0,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = load_config().get("inference", {})
        self.model_name = cfg.get("model", model_name)
        self.max_new_tokens = int(cfg.get("max_new_tokens", max_new_tokens))
        self.temperature = float(cfg.get("temperature", temperature))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if device:
            kwargs["device_map"] = device
        elif torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **kwargs,
        )
        self.model.eval()
        self.system_prompt = load_prompt()

    def _chat(self, user: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = f"{self.system_prompt}\n\nUSER:\n{user}\n\nASSISTANT:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text

    def label_note(self, text: str, concepts: list[str], max_attempts: int = 5) -> list[dict]:
        user = (
            "REQUESTED CONCEPTS: "
            + ", ".join(sorted(concepts))
            + "\n\nTEXT:\n"
            + text
        )
        last_err: Exception | None = None
        for attempt in range(max_attempts):
            try:
                raw = self._chat(user)
                return _parse_labels(raw, concepts, text)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(min(2 ** attempt, 16))
        raise RuntimeError(f"labeling failed after {max_attempts} attempts") from last_err


def group_notes_by_hadm(source: Path | None = None) -> dict[int, dict]:
    from data_generation.silentehr import load_rows  # noqa: WPS433

    rows = load_rows(source or SOURCE_STRUCT)
    grouped: dict[int, dict] = {}
    for row in rows:
        item = grouped.setdefault(row["hadm_id"], {
            "hadm_id": row["hadm_id"],
            "text": row["text"],
            "concepts": [],
        })
        if row["concept"] not in item["concepts"]:
            item["concepts"].append(row["concept"])
    return grouped


def relabel_mimic_notes(
    limit: int = 0,
    output: Path | None = None,
    source: Path | None = None,
    device: str | None = None,
) -> Path:
    output = output or LABELS_PATH
    grouped = group_notes_by_hadm(source)
    completed: set[int] = set()
    if output.exists():
        for line in output.open(encoding="utf-8"):
            try:
                completed.add(json.loads(line)["hadm_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    pending = [
        item for hadm, item in sorted(grouped.items())
        if hadm not in completed
    ]
    if limit > 0:
        pending = pending[:limit]
    print(
        f"Qwen3-12B relabel: {len(grouped)} unique notes, "
        f"{len(completed)} resumed, {len(pending)} pending",
        flush=True,
    )
    labeler = Qwen3ObservationLabeler(device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_fail = 0
    with output.open("a", encoding="utf-8") as handle:
        for item in pending:
            try:
                labels = labeler.label_note(item["text"], item["concepts"])
                record = {
                    "hadm_id": item["hadm_id"],
                    "labels": labels,
                    "model": OBSERVATION_LABEL_MODEL,
                    "label_model": labeler.model_name,
                    "labeled_at": datetime.now().isoformat(),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                print(f"  failed hadm={item['hadm_id']}: {exc}", flush=True)
    summary = {
        "model": OBSERVATION_LABEL_MODEL,
        "label_model": labeler.model_name,
        "unique_notes": len(grouped),
        "newly_labeled": n_ok,
        "failures": n_fail,
        "output": str(output),
        "finished_at": datetime.now().isoformat(),
    }
    summary_path = OUTPUT_DIR / "qwen3_relabel_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return output


def build_qwen3_struct(
    source: Path | None = None,
    labels_path: Path | None = None,
    output: Path | None = None,
) -> Path:
    from collections import Counter

    from data_generation.silentehr import load_rows  # noqa: WPS433

    source = source or SOURCE_STRUCT
    labels_path = labels_path or LABELS_PATH
    output = output or STRUCT_PATH
    if not source.exists():
        raise FileNotFoundError(source)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    qwen_labels: dict[tuple[int, str], dict] = {}
    for line in labels_path.open(encoding="utf-8"):
        obj = json.loads(line)
        for lab in obj["labels"]:
            qwen_labels[(obj["hadm_id"], lab["concept"])] = lab

    rows_out = []
    missing = 0
    for row in load_rows(source):
        lab = qwen_labels.get((row["hadm_id"], row["concept"]))
        if lab is None:
            missing += 1
            continue
        status = lab["observation_status"]
        if status not in QWEN3_TO_MENTION:
            raise ValueError(f"unknown Qwen3 status: {status}")
        updated = dict(row)
        updated["mention_status"] = QWEN3_TO_MENTION[status]
        updated["qwen3_observation_status"] = status
        updated["qwen3_evidence_span"] = lab.get("evidence_span", "")
        updated["observation_label_source"] = OBSERVATION_LABEL_MODEL
        rows_out.append(updated)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    cohort = Counter()
    for row in rows_out:
        if row["mention_status"] in (
                "observed_mixed", "observed_uncertain", "observed_historical"):
            continue
        cohort[(row["concept"], row["mention_status"],
                "struct_pos" if row["structured_positive"] else "struct_neg")] += 1
    print(f"wrote {len(rows_out)} Qwen3-labeled rows -> {output}")
    if missing:
        print(f"  skipped {missing} rows without Qwen3 labels")
    return output


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Qwen3-12B MIMIC observation labeling")
    ap.add_argument("cmd", choices=["label", "build-struct"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    ensure_repo_on_path()
    if args.cmd == "label":
        relabel_mimic_notes(limit=args.limit, device=args.device)
    else:
        build_qwen3_struct()
