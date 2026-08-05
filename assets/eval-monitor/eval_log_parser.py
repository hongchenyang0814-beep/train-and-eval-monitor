from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from typing import Any


TASK_CATALOG = [
    ("challenge_itemic_pattern_caption_video", "懂物料", "视频物料理解"),
    ("challenge_itemic_pattern_caption_product", "懂物料", "商品物料理解"),
    ("challenge_itemic_pattern_caption_ad", "懂物料", "广告物料理解"),
    ("challenge_itemic_pattern_caption_live", "懂物料", "直播物料理解"),
    ("challenge_evolution_action_select", "懂用户", "用户行为选择"),
    ("challenge_evolution_topic_gen", "懂用户", "用户主题生成"),
    ("challenge_recommendation_video", "懂推荐", "视频推荐"),
    ("challenge_recommendation_product", "懂推荐", "商品推荐"),
    ("challenge_recommendation_ad", "懂推荐", "广告推荐"),
    ("challenge_recommendation_live", "懂推荐", "直播推荐"),
    ("challenge_common_sense", "常识", "通用常识"),
]

TASK_INFO = {key: {"group": group, "label": label} for key, group, label in TASK_CATALOG}
PARSER_VERSION = 3

ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
RICH_TAG_RE = re.compile(
    r"\[/?(?:black|red|green|yellow|blue|magenta|cyan|white|bold|dim|italic|underline|blink|reverse|strike)(?:\s+[^]]+)?\]",
    re.IGNORECASE,
)
TASK_RE = re.compile(r"Task\s*\[\d+\s*/\s*\d+\]\s*:\s*([\w.-]+)")
TIMESTAMP_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})(?:,\d+)?\]")
NUMBER_RE = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
SID_TRIPLE_RE = re.compile(r"<s_a_\d+>\s*<s_b_\d+>\s*<s_c_\d+>", re.I)
ANALYSIS_MARKER = "########## 评分与自动分析指标 ##########"


def decode_log_bytes(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def clean_line(line: str) -> str:
    line = ANSI_RE.sub("", line)
    line = RICH_TAG_RE.sub("", line)
    line = line.replace("\x00", "")
    return line.strip()


def _new_task(key: str) -> dict[str, Any]:
    info = TASK_INFO.get(key, {"group": "其他", "label": key})
    return {
        "key": key,
        "group": info["group"],
        "label": info["label"],
        "status": "未发现",
        "evaluator": "",
        "sample_count": None,
        "generation_seconds": None,
        "seconds_per_sample": None,
        "matching": None,
        "sid_mapping_count": None,
        "metrics": {},
        "result_path": "",
        "debug_path": "",
        "example_count": 0,
    }


def _is_progress(line: str) -> bool:
    return bool(
        re.search(
            r"(?:Processed prompts|Generating|Extracting Model WIPs|Batches|SID eval|PID eval|Loading safetensors).*\d+%\|",
            line,
            re.IGNORECASE,
        )
    )


def _noise_kind(line: str) -> str | None:
    lower = line.lower()
    if not line:
        return "blank"
    if _is_progress(line):
        return "progress"
    if "http request:" in lower:
        return "http"
    if "syntaxwarning:" in lower or "invalid escape sequence" in lower:
        return "startup_warning"
    if any(
        token in lower
        for token in (
            "loading safetensors",
            "enginecore_dp",
            "initializing a v1 llm engine",
            "starting to load model",
            "torch.distributed",
            "rank 0 is connected",
            "cuda graph",
            "cudagraph",
            "compilation_config",
            "non-default args:",
        )
    ):
        return "runtime"
    if re.match(r"^[│┃╭╰┏┗┌└─━═\s]+$", line):
        return "decoration"
    return None


def _parse_json_sample(line: str, task: str, index: int) -> dict[str, Any] | None:
    if len(line) > 30000 or not line.startswith("{") or not line.endswith("}"):
        return None
    try:
        obj = json.loads(line)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    aliases = {
        "prompt": ("prompt", "question", "input", "instruction", "query"),
        "response": ("response", "prediction", "model_output", "output", "generated_text"),
        "reference": ("reference", "ground_truth", "target", "answer", "label"),
    }
    sample: dict[str, Any] = {"id": str(obj.get("id", obj.get("sample_id", index))), "task": task}
    for target, keys in aliases.items():
        for key in keys:
            if key in obj and obj[key] is not None:
                value = obj[key]
                sample[target] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                break
    if "prompt" not in sample or "response" not in sample:
        return None
    metrics = obj.get("metrics", obj.get("score"))
    if metrics is not None:
        sample["metrics"] = metrics
    return sample


def _extract_samples(lines: list[str], task_at_line: list[str]) -> list[dict[str, Any]]:
    detailed_starts = [idx for idx, line in enumerate(lines) if re.match(r"^Sample ID\s*:", line, re.I)]
    if detailed_starts:
        generation_results: list[tuple[int, str]] = []
        for idx, line in enumerate(lines):
            if "Generation results saved to:" not in line:
                continue
            nearby = " ".join(lines[idx : min(len(lines), idx + 4)])
            task_match = re.search(r"/(challenge_[\w.-]+)/", nearby)
            if task_match:
                generation_results.append((idx, task_match.group(1)))

        samples: list[dict[str, Any]] = []
        for position, start in enumerate(detailed_starts):
            marker = re.match(r"^Sample ID\s*:\s*(.+?)\s*$", lines[start], re.I)
            sample_id = marker.group(1) if marker else str(position)
            next_start = detailed_starts[position + 1] if position + 1 < len(detailed_starts) else len(lines)
            end = next_start
            for idx in range(start + 1, next_start):
                if (
                    re.match(r"^={20,}$", lines[idx])
                    or re.match(r"^Total time:\s*[\d.]+s", lines[idx], re.I)
                    or "Generation results saved to:" in lines[idx]
                    or TASK_RE.search(lines[idx])
                ):
                    end = idx
                    break

            task = next((task for result_idx, task in generation_results if result_idx > start), task_at_line[start])
            sample: dict[str, Any] = {"id": sample_id, "task": task, "variants": {}}
            active_mode = ""
            active_target: list[str] | None = None

            for idx in range(start + 1, end):
                line = lines[idx]
                input_match = re.match(r"^(Think|No-think)\s+Input\s*:\s*(.*)$", line, re.I)
                if input_match:
                    active_mode = "think" if input_match.group(1).lower() == "think" else "no_think"
                    variant = sample["variants"].setdefault(
                        active_mode, {"mode": active_mode, "input": "", "outputs": []}
                    )
                    variant["_input_lines"] = [input_match.group(2)] if input_match.group(2) else []
                    active_target = variant["_input_lines"]
                    continue

                output_match = re.match(r"^(Think|No-think)\s+Output\[(\d+)\]\s*:\s*(.*)$", line, re.I)
                if output_match:
                    active_mode = "think" if output_match.group(1).lower() == "think" else "no_think"
                    variant = sample["variants"].setdefault(
                        active_mode, {"mode": active_mode, "input": "", "outputs": []}
                    )
                    output = {
                        "index": int(output_match.group(2)),
                        "text": "",
                        "_lines": [output_match.group(3)] if output_match.group(3) else [],
                    }
                    variant["outputs"].append(output)
                    active_target = output["_lines"]
                    continue

                if active_target is not None and _noise_kind(line) not in {
                    "progress",
                    "http",
                    "startup_warning",
                    "runtime",
                    "decoration",
                }:
                    active_target.append(line)

            for variant in sample["variants"].values():
                variant["input"] = "\n".join(variant.pop("_input_lines", [])).strip()
                for output in variant["outputs"]:
                    output["text"] = "\n".join(output.pop("_lines", [])).strip()
                variant["outputs"].sort(key=lambda item: item["index"])

            preferred = sample["variants"].get("no_think") or sample["variants"].get("think")
            if preferred:
                sample["prompt"] = preferred["input"]
                sample["response"] = preferred["outputs"][0]["text"] if preferred["outputs"] else ""
            samples.append(sample)
        return samples

    samples: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    active_field = ""
    sample_marker = re.compile(r"^(?:\[)?(?:Sample|Example)\s*(?:#|ID|Index)?\s*[:=]?\s*([\w.-]+)", re.I)
    field_marker = re.compile(
        r"^(Prompt|Question|Input|Instruction|Query|Model Output|Response|Prediction|Generated Text|Reference|Ground Truth|Target|Answer|Label|Score|Metrics)\s*[:=]\s*(.*)$",
        re.I,
    )
    field_map = {
        "prompt": "prompt",
        "question": "prompt",
        "input": "prompt",
        "instruction": "prompt",
        "query": "prompt",
        "model output": "response",
        "response": "response",
        "prediction": "response",
        "generated text": "response",
        "reference": "reference",
        "ground truth": "reference",
        "target": "reference",
        "answer": "reference",
        "label": "reference",
        "score": "metrics",
        "metrics": "metrics",
    }

    def commit() -> None:
        nonlocal current
        if current and current.get("prompt") and current.get("response"):
            for key in ("prompt", "response", "reference"):
                if isinstance(current.get(key), str):
                    current[key] = current[key].strip()[:12000]
            samples.append(current)
        current = None

    for idx, line in enumerate(lines):
        if len(samples) >= 300:
            break
        parsed = _parse_json_sample(line, task_at_line[idx], len(samples) + 1)
        if parsed:
            commit()
            samples.append(parsed)
            active_field = ""
            continue
        marker = sample_marker.match(line)
        if marker and "sample size" not in line.lower():
            commit()
            current = {"id": marker.group(1), "task": task_at_line[idx]}
            active_field = ""
            continue
        field = field_marker.match(line)
        if field:
            mapped = field_map[field.group(1).lower()]
            if current is None:
                current = {"id": str(len(samples) + 1), "task": task_at_line[idx]}
            current[mapped] = field.group(2).strip()
            active_field = mapped
            continue
        if current and active_field and line and not TASK_RE.search(line):
            if len(str(current.get(active_field, ""))) < 12000:
                current[active_field] = f"{current.get(active_field, '')}\n{line}".strip()
        elif current and (TASK_RE.search(line) or line.startswith("Updated sample metrics")):
            commit()
            active_field = ""
    commit()
    return samples


def _render_sample_export(samples: list[dict[str, Any]]) -> list[str]:
    output: list[str] = []
    task_order = {key: idx for idx, (key, _, _) in enumerate(TASK_CATALOG)}
    ordered = sorted(samples, key=lambda sample: (task_order.get(sample.get("task", ""), 999), str(sample.get("id", ""))))
    current_task = ""
    for sample in ordered:
        task = sample.get("task", "")
        if task != current_task:
            info = TASK_INFO.get(task, {"label": task})
            output.extend(["", f"########## {info['label']} ({task}) 示例样本 ##########"])
            current_task = task
        output.extend(["", "============================================================", f"Sample ID: {sample.get('id', '')}"])
        variants = sample.get("variants") or {}
        for mode in ("think", "no_think"):
            variant = variants.get(mode)
            if not variant:
                continue
            label = "Think" if mode == "think" else "No-think"
            output.append(f"{label} Input:")
            output.extend(str(variant.get("input", "")).splitlines())
            for item in variant.get("outputs", []):
                output.append(f"{label} Output[{item.get('index', 0)}]:")
                output.extend(str(item.get("text", "")).splitlines())
    return output


def _final_answer_text(value: str) -> str:
    text = str(value or "").strip()
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.I)[-1]
    return text.strip()


def _sid_candidates(value: str) -> list[str]:
    return [re.sub(r"\s+", "", match.group(0)).lower() for match in SID_TRIPLE_RE.finditer(str(value or ""))]


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _variant_sid_metrics(variant: dict[str, Any], historical: set[str]) -> dict[str, Any]:
    output_count = 0
    candidate_count = 0
    direct_copy_count = 0
    repeated_sid_count = 0
    repeated_output_count = 0
    unique_candidates: set[str] = set()
    for output in variant.get("outputs", []):
        output_count += 1
        candidates = _sid_candidates(_final_answer_text(output.get("text", "")))
        candidate_count += len(candidates)
        unique_candidates.update(candidates)
        direct_copy_count += sum(1 for candidate in candidates if candidate in historical)
        duplicate_count = len(candidates) - len(set(candidates))
        repeated_sid_count += duplicate_count
        repeated_output_count += int(duplicate_count > 0)
    return {
        "output_count": output_count,
        "candidate_count": candidate_count,
        "unique_candidate_count": len(unique_candidates),
        "direct_copy_count": direct_copy_count,
        "direct_copy_rate": _rate(direct_copy_count, candidate_count),
        "repeated_sid_count": repeated_sid_count,
        "repeated_output_count": repeated_output_count,
        "repeat_rate": _rate(repeated_sid_count, candidate_count),
        "historical_sid_count": len(historical),
        "hallucination_count": max(0, candidate_count - direct_copy_count),
        "hallucination_rate": _rate(max(0, candidate_count - direct_copy_count), candidate_count),
    }


def _compute_automatic_metrics(task_list: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_task.setdefault(str(sample.get("task", "")), []).append(sample)

    metrics_by_task: dict[str, dict[str, Any]] = {}
    for task in task_list:
        key = task["key"]
        task_samples = by_task.get(key, [])
        output_count = 0
        sid_count = 0
        unique_sids: set[str] = set()
        variant_count = 0
        copy_by_mode: dict[str, dict[str, Any]] = {}
        for sample in task_samples:
            variants = sample.get("variants") or {}
            for mode, variant in variants.items():
                variant_count += 1
                historical = set(_sid_candidates(variant.get("input", "")))
                stats = _variant_sid_metrics(variant, historical)
                existing = copy_by_mode.setdefault(
                    mode,
                    {
                        "output_count": 0,
                        "candidate_count": 0,
                        "unique_candidate_count": 0,
                        "direct_copy_count": 0,
                        "repeated_sid_count": 0,
                        "repeated_output_count": 0,
                        "hallucination_count": 0,
                        "historical_sid_count": 0,
                        "_unique": set(),
                    },
                )
                for field in (
                    "output_count",
                    "candidate_count",
                    "direct_copy_count",
                    "repeated_sid_count",
                    "repeated_output_count",
                    "hallucination_count",
                    "historical_sid_count",
                ):
                    existing[field] += stats[field]
                existing["_unique"].update(
                    _sid_candidates("\n".join(str(item.get("text", "")) for item in variant.get("outputs", [])))
                )
                output_count += stats["output_count"]
                sid_count += stats["candidate_count"]
                unique_sids.update(existing["_unique"])

        for stats in copy_by_mode.values():
            stats["unique_candidate_count"] = len(stats.pop("_unique"))
            stats["direct_copy_rate"] = _rate(stats["direct_copy_count"], stats["candidate_count"])
            stats["repeat_rate"] = _rate(stats["repeated_sid_count"], stats["candidate_count"])
            stats["hallucination_rate"] = _rate(stats["hallucination_count"], stats["candidate_count"])
            stats["sample_count"] = len(task_samples)

        result: dict[str, Any] = {
            "sample_count": len(task_samples),
            "variant_count": variant_count,
            "output_count": output_count,
            "sid_candidate_count": sid_count,
            "unique_sid_count": len(unique_sids),
        }
        if key.startswith("challenge_recommendation_"):
            overlap_count = 0
            union_count = 0
            paired_output_count = 0
            same_output_count = 0
            for sample in task_samples:
                variants = sample.get("variants") or {}
                think = variants.get("think")
                no_think = variants.get("no_think")
                if not think or not no_think:
                    continue
                think_by_index = {int(item.get("index", 0)): set(_sid_candidates(_final_answer_text(item.get("text", "")))) for item in think.get("outputs", [])}
                no_think_by_index = {int(item.get("index", 0)): set(_sid_candidates(_final_answer_text(item.get("text", "")))) for item in no_think.get("outputs", [])}
                for index in sorted(set(think_by_index) | set(no_think_by_index)):
                    left = think_by_index.get(index, set())
                    right = no_think_by_index.get(index, set())
                    paired_output_count += 1
                    overlap_count += len(left & right)
                    union_count += len(left | right)
                    same_output_count += int(left == right and bool(left or right))
            result["copy_answer"] = copy_by_mode
            result["think_no_think_overlap"] = {
                "paired_output_count": paired_output_count,
                "overlap_sid_count": overlap_count,
                "union_sid_count": union_count,
                "overlap_rate": _rate(overlap_count, union_count),
                "same_output_count": same_output_count,
                "same_output_rate": _rate(same_output_count, paired_output_count),
            }
        elif key == "challenge_evolution_action_select":
            result["hallucination"] = copy_by_mode
            result["answer_repeat"] = copy_by_mode
        elif copy_by_mode:
            result["sid_analysis"] = copy_by_mode
        metrics_by_task[key] = result
    return metrics_by_task


def _render_analysis_appendix(parsed: dict[str, Any], score: dict[str, Any] | None = None) -> str:
    score = score or {}
    lines = [ANALYSIS_MARKER, f"人工总分: {score.get('total') if score.get('total') not in (None, '') else '未录入'}"]
    task_scores = score.get("tasks") or {}
    for task in parsed.get("tasks", []):
        key = task.get("key", "")
        info = TASK_INFO.get(key, {"label": task.get("label", key)})
        manual = task_scores.get(key)
        auto = (parsed.get("automatic_metrics") or {}).get(key, {})
        lines.append(f"任务: {info['label']} ({key})")
        lines.append(f"  人工评分: {manual if manual not in (None, '') else '未录入'}")
        lines.append(f"  展示样本: {auto.get('sample_count', task.get('example_count', 0))}")
        if key.startswith("challenge_recommendation_"):
            for mode, label in (("think", "Think"), ("no_think", "No-think")):
                metric = (auto.get("copy_answer") or {}).get(mode)
                if metric:
                    lines.append(
                        f"  抄答案指标 {label}: {metric['direct_copy_count']}/{metric['candidate_count']} "
                        f"({metric['direct_copy_rate']:.2%})"
                    )
            overlap = auto.get("think_no_think_overlap") or {}
            if overlap:
                lines.append(
                    f"  Think/No-think SID重复: {overlap['overlap_sid_count']}/{overlap['union_sid_count']} "
                    f"({overlap['overlap_rate']:.2%})"
                )
        if key == "challenge_evolution_action_select":
            metric = (auto.get("hallucination") or {}).get("no_think")
            if metric:
                lines.append(
                    f"  幻觉情况 No-think: {metric['hallucination_count']}/{metric['candidate_count']} "
                    f"({metric['hallucination_rate']:.2%})"
                )
                lines.append(
                    f"  重复情况 No-think: {metric['repeated_sid_count']} 个重复SID，"
                    f"{metric['repeated_output_count']} 个输出出现循环"
                )
        lines.append("")
    return "\n".join(lines).rstrip()


def render_filtered_text(parsed: dict[str, Any], score: dict[str, Any] | None = None) -> str:
    base = str(parsed.get("filtered_text", ""))
    if ANALYSIS_MARKER in base:
        base = base.split(ANALYSIS_MARKER, 1)[0].rstrip()
    return f"{base}\n\n{_render_analysis_appendix(parsed, score)}\n"


def parse_log(text: str, filename: str, file_size: int | None = None) -> dict[str, Any]:
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [clean_line(line) for line in raw_lines]
    counters: Counter[str] = Counter()
    tasks: dict[str, dict[str, Any]] = {key: _new_task(key) for key, _, _ in TASK_CATALOG}
    configured_tasks: list[str] = []
    metadata: dict[str, Any] = {}
    issues: list[dict[str, str]] = []
    filtered: list[str] = []
    current_task = ""
    task_at_line: list[str] = []
    pending_path: tuple[str, str] | None = None
    timestamps: list[str] = []
    total_seconds: float | None = None
    generation_phase: dict[str, bool] = {}
    generation_chunks: dict[str, dict[str, Any]] = {}

    def keep(value: str) -> None:
        value = value.strip()
        if value and (not filtered or filtered[-1] != value):
            filtered.append(value)

    def ensure_task(key: str) -> dict[str, Any]:
        if key not in tasks:
            tasks[key] = _new_task(key)
        return tasks[key]

    for line in lines:
        task_at_line.append(current_task)
        noise = _noise_kind(line)
        if noise:
            counters[noise] += 1
        else:
            counters["other"] += 1

        for stamp in TIMESTAMP_RE.findall(line):
            timestamps.append(stamp)

        config = re.search(
            r"\[INFO\]\s*(model_path|version|data_dir|output_dir|tasks|thinking|race_mode|overwrite|seed)\s*:\s*(.+)$",
            line,
            re.I,
        )
        if config:
            key, value = config.group(1).lower(), config.group(2).strip()
            metadata[key] = value
            keep(line)
            if key == "tasks":
                try:
                    parsed_tasks = ast.literal_eval(value)
                    if isinstance(parsed_tasks, list):
                        configured_tasks = [str(item) for item in parsed_tasks]
                except (ValueError, SyntaxError):
                    configured_tasks = re.findall(r"challenge_[\w.-]+", value)
            continue

        task_match = TASK_RE.search(line)
        if task_match:
            current_task = task_match.group(1)
            task_at_line[-1] = current_task
            task = ensure_task(current_task)
            if task["status"] == "未发现":
                task["status"] = "运行中/未完成"
            generation_phase.setdefault(current_task, True)
            keep(f"\n===== {task['label']} ({current_task}) =====")
            continue

        task = ensure_task(current_task) if current_task else None

        if pending_path and line.startswith(("/", "C:\\", "D:\\")):
            task_key, field = pending_path
            ensure_task(task_key)[field] = line
            keep(line)
            pending_path = None
            continue

        evaluator = re.search(r"Using\s+([\w.]+Evaluator)\s+for\s+([\w.-]+)", line)
        if evaluator:
            current_task = evaluator.group(2)
            task_at_line[-1] = current_task
            task = ensure_task(current_task)
            task["evaluator"] = evaluator.group(1)
            task["status"] = "评测中"
            generation_phase[current_task] = False
            keep(f"Using {evaluator.group(1)} for {current_task}")
            continue

        progress_count = re.search(r"Processed prompts:\s*(\d+)%.*?\|\s*(\d+)\s*/\s*(\d+)\s*\[", line)
        if progress_count and task and generation_phase.get(current_task, True):
            percent, current, total = map(int, progress_count.groups())
            chunk = generation_chunks.setdefault(current_task, {"open": False, "counted": False, "total": 0, "sum": 0})
            if percent == 0 or current == 0:
                chunk.update({"open": True, "counted": False, "total": total})
            elif not chunk["open"]:
                chunk.update({"open": True, "counted": False, "total": total})
            if current == total and not chunk["counted"]:
                chunk["sum"] += total
                chunk["counted"] = True
                task["sample_count"] = chunk["sum"]

        generation = re.search(
            rf"Total time:\s*({NUMBER_RE})s,\s*Average per sample:\s*({NUMBER_RE})s", line, re.I
        )
        if generation and task:
            task["generation_seconds"] = float(generation.group(1))
            task["seconds_per_sample"] = float(generation.group(2))
            keep(f"Total time: {generation.group(1)}s, Average per sample: {generation.group(2)}s")
            continue

        matching = re.search(r"Matching statistics:\s*(\d+) attempted,\s*(\d+) parsed,\s*(\d+) valid", line, re.I)
        if matching and task:
            task["matching"] = {
                "attempted": int(matching.group(1)),
                "parsed": int(matching.group(2)),
                "valid": int(matching.group(3)),
            }
            keep(
                f"Matching statistics: {matching.group(1)} attempted, "
                f"{matching.group(2)} parsed, {matching.group(3)} valid"
            )
            continue

        sid_mapping = re.search(r"Loaded\s+(\d+)\s+SID mappings", line, re.I)
        if sid_mapping and task:
            task["sid_mapping_count"] = int(sid_mapping.group(1))
            keep(f"Loaded {sid_mapping.group(1)} SID mappings")
            continue

        metric = re.search(
            rf"(Macro\s+(?:Unweighted|Importance-weighted|Double-weighted)\s+F1)\s*:\s*({NUMBER_RE})(?:\s*\(Core:\s*({NUMBER_RE})\))?",
            line,
            re.I,
        )
        if metric and task:
            metric_name = re.sub(r"\s+", " ", metric.group(1)).title().replace("F1", "F1")
            task["metrics"][metric_name] = float(metric.group(2))
            if metric.group(3) is not None:
                task["metrics"][f"{metric_name} Core"] = float(metric.group(3))
            keep(metric.group(0))
            continue

        generic_metric = re.search(rf"^([A-Za-z][A-Za-z0-9 _/().-]{{2,60}})\s*:\s*({NUMBER_RE})\s*$", line)
        if generic_metric and task and re.search(
            r"accuracy|precision|recall|f1|bleu|rouge|ndcg|mrr|similarity|validity|pass rate|exact match",
            generic_metric.group(1),
            re.I,
        ):
            task["metrics"][generic_metric.group(1).strip()] = float(generic_metric.group(2))
            keep(generic_metric.group(0))
            continue

        if "Updated sample metrics to:" in line and task:
            task["status"] = "已完成"
            keep("Updated sample metrics to:")
            tail = line.split("Updated sample metrics to:", 1)[1].strip()
            if tail.startswith(("/", "C:\\", "D:\\")):
                task["result_path"] = tail
                keep(tail)
            else:
                pending_path = (current_task, "result_path")
            continue

        if "Created debug file:" in line and task:
            keep("Created debug file:")
            tail = line.split("Created debug file:", 1)[1].strip()
            if tail.startswith(("/", "C:\\", "D:\\")):
                task["debug_path"] = tail
                keep(tail)
            else:
                pending_path = (current_task, "debug_path")
            continue

        global_time = re.search(r"Total time:\s*([\d.]+)s\s*\(([\d.]+)min\)", line, re.I)
        if global_time:
            total_seconds = float(global_time.group(1))
            keep(f"Total time: {global_time.group(1)}s ({global_time.group(2)}min)")
            continue

        if re.search(r"^(?:✓\s*)?Results Saved to|Aggregating L1-L4|eval_results:|report:|done in|Competition.*(?:result|metrics)", line, re.I):
            keep(line)
            continue

        if re.search(r"\[(?:ERROR|WARNING)\]|Traceback|\bException\b|CUDA out of memory|failed", line, re.I):
            if "SyntaxWarning" not in line and "invalid escape sequence" not in line:
                if re.search(r"Failed tasks:\s*0\b", line, re.I):
                    keep(line)
                    continue
                level = "error" if re.search(r"\[ERROR\]|Traceback|Exception|out of memory", line, re.I) else "warning"
                if len(issues) < 50:
                    issues.append({"level": level, "text": line[:600], "task": current_task})
                keep(line)
            continue

        if re.match(r"^(?:Prompt|Question|Input|Instruction|Model Output|Response|Prediction|Reference|Ground Truth|Target|Answer|Score|Metrics)\s*[:=]", line, re.I):
            keep(line[:12000])

    for key in configured_tasks:
        ensure_task(key)

    task_list = list(tasks.values())
    catalog_order = {key: idx for idx, (key, _, _) in enumerate(TASK_CATALOG)}
    task_list.sort(key=lambda item: (catalog_order.get(item["key"], 999), item["key"]))
    samples = _extract_samples(lines, task_at_line)
    sample_counts = Counter(sample.get("task", "") for sample in samples)
    for task in task_list:
        task["example_count"] = sample_counts.get(task["key"], 0)
    automatic_metrics = _compute_automatic_metrics(task_list, samples)
    for task in task_list:
        task["automatic_metrics"] = automatic_metrics.get(task["key"], {"sample_count": task["example_count"]})
    sample_export = _render_sample_export(samples)

    started_at = timestamps[0] if timestamps else ""
    ended_at = timestamps[-1] if timestamps else ""
    if total_seconds is None and started_at and ended_at:
        try:
            start_dt = datetime.strptime(started_at, "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(ended_at, "%Y-%m-%d %H:%M:%S")
            total_seconds = max(0.0, (end_dt - start_dt).total_seconds())
        except ValueError:
            pass

    nonempty_count = sum(1 for line in lines if line)
    retained_count = sum(1 for line in filtered + sample_export if line.strip())
    completed_count = sum(1 for task in task_list if task["status"] == "已完成")
    metric_count = sum(len(task["metrics"]) for task in task_list)
    filtered_header = [
        "评测日志精简导出",
        f"来源文件: {filename}",
        f"原始行数: {nonempty_count}",
        f"保留信息行: {retained_count}",
        "说明: 已移除模型加载、HTTP 请求、进度条、ANSI 控制符和重复运行时信息。",
        "",
    ]
    result = {
        "parser_version": PARSER_VERSION,
        "source": {
            "filename": filename,
            "file_size": file_size if file_size is not None else len(text.encode("utf-8", errors="replace")),
            "sha256": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": total_seconds,
        },
        "metadata": metadata,
        "summary": {
            "task_count": len([task for task in task_list if task["status"] != "未发现"]),
            "completed_count": completed_count,
            "metric_count": metric_count,
            "sample_detail_count": len(samples),
            "raw_line_count": nonempty_count,
            "retained_line_count": retained_count,
            "retained_ratio": round(retained_count / nonempty_count, 4) if nonempty_count else 0,
            "issue_count": len(issues),
        },
        "noise": dict(counters),
        "tasks": task_list,
        "automatic_metrics": automatic_metrics,
        "samples": samples,
        "issues": issues,
        "filtered_text": "\n".join(filtered_header + filtered + sample_export).strip() + "\n",
    }
    result["filtered_text"] = render_filtered_text(result)
    return result
