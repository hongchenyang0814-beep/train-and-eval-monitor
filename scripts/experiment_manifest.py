#!/usr/bin/env python3
"""Create, validate, and render the required AI training-task explanation document."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
REQUIRED_FIELDS = (
    "schema_version", "run_id", "title", "purpose", "hypothesis", "changes",
    "dataset", "model", "config_file", "expected_result", "notes", "created_at",
)


def template(output_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": output_dir.name,
        "title": "填写本次训练任务标题",
        "purpose": "填写为什么执行本次训练",
        "hypothesis": "填写希望验证的假设",
        "changes": ["填写相对基线的改动；没有改动也要明确写出"],
        "comparison_run": "填写基线任务 ID；没有基线写 none",
        "dataset": {
            "summary": "填写数据集、切分和规模",
            "sources": ["填写数据来源或版本"],
        },
        "model": {
            "base_model": "填写基础模型 ID",
            "training_method": "填写 LoRA、SFT 或其他方法",
        },
        "config_file": "training_config.yaml",
        "config_summary": [{"label": "填写关键参数", "value": "填写值和选择理由"}],
        "expected_result": "填写预期指标变化或验收条件",
        "notes": "填写风险、限制、后续动作和异常说明",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"manifest 不存在: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"manifest 无法读取: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("manifest 顶层必须是 JSON 对象")
    return value


def validate(value: dict[str, Any], output_dir: Path) -> list[str]:
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in value or value[field] in (None, "", [], {}):
            errors.append(f"缺少必填字段: {field}")
    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version 必须为 {SCHEMA_VERSION}")
    run_id = str(value.get("run_id", ""))
    if not RUN_ID_RE.fullmatch(run_id):
        errors.append("run_id 只能包含字母、数字、下划线和连字符")
    changes = value.get("changes")
    if not isinstance(changes, list) or not all(isinstance(item, str) and item.strip() for item in changes):
        errors.append("changes 必须是非空字符串数组")
    for field in ("dataset", "model"):
        if not isinstance(value.get(field), dict):
            errors.append(f"{field} 必须是对象")
    config_file = value.get("config_file")
    if not isinstance(config_file, str) or not config_file.strip() or Path(config_file).name != config_file:
        errors.append("config_file 必须是任务目录内的文件名，不能是绝对路径")
    elif not (output_dir / config_file).is_file():
        errors.append(f"找不到配置文件: {output_dir / config_file}")
    return errors


def render(value: dict[str, Any]) -> str:
    dataset = value.get("dataset") if isinstance(value.get("dataset"), dict) else {}
    model = value.get("model") if isinstance(value.get("model"), dict) else {}
    lines = [
        f"# {value.get('title', value.get('run_id', '训练任务'))}",
        "",
        f"- 任务 ID：`{value.get('run_id', '')}`",
        f"- 创建时间：{value.get('created_at', '')}",
        "",
        "## 目的与假设",
        "",
        str(value.get("purpose", "")),
        "",
        f"验证假设：{value.get('hypothesis', '')}",
        "",
        "## 相对基线的改动",
        "",
    ]
    lines.extend(f"- {item}" for item in value.get("changes", []))
    lines.extend([
        "",
        f"基线任务：{value.get('comparison_run', 'none')}",
        "",
        "## 数据与模型",
        "",
        f"- 数据摘要：{dataset.get('summary', '')}",
        f"- 数据来源：{', '.join(str(item) for item in dataset.get('sources', []))}",
        f"- 基础模型：{model.get('base_model', '')}",
        f"- 训练方法：{model.get('training_method', '')}",
        f"- 配置文件：`{value.get('config_file', '')}`",
        "",
        "## 预期结果",
        "",
        str(value.get("expected_result", "")),
        "",
        "## 备注与风险",
        "",
        str(value.get("notes", "")),
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "validate", "render"):
        item = sub.add_parser(command)
        item.add_argument("--output-dir", required=True, type=Path)
        item.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "run_manifest.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.command == "init":
            if manifest_path.exists() and not args.force:
                raise ValueError(f"manifest 已存在，使用 --force 才能覆盖: {manifest_path}")
            manifest_path.write_text(json.dumps(template(output_dir), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (output_dir / "training_task.md").write_text(render(load(manifest_path)), encoding="utf-8")
            print(json.dumps({"status": "created", "manifest": str(manifest_path), "document": str(output_dir / 'training_task.md')}, ensure_ascii=False))
            return 0
        value = load(manifest_path)
        errors = validate(value, output_dir)
        if errors:
            raise ValueError("; ".join(errors))
        if args.command == "render":
            (output_dir / "training_task.md").write_text(render(value), encoding="utf-8")
            print(json.dumps({"status": "rendered", "document": str(output_dir / 'training_task.md')}, ensure_ascii=False))
        else:
            print(json.dumps({"status": "valid", "manifest": str(manifest_path)}, ensure_ascii=False))
        return 0
    except ValueError as error:
        print(json.dumps({"status": "invalid", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
