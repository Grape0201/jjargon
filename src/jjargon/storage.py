import json
from pathlib import Path

import yaml

from jjargon.models import Glossary


def load_glossary(path: Path) -> Glossary:
    """YAMLまたはJSONファイルから専門用語集(Glossary)を読み込む。存在しない場合は空のGlossaryを返す。"""
    if not path.exists():
        return Glossary()

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return Glossary()

    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(content)
    else:
        data = yaml.safe_load(content)

    if data is None:
        return Glossary()
    if isinstance(data, list):
        return Glossary.model_validate({"terms": data})
    return Glossary.model_validate(data)


def save_glossary(glossary: Glossary, path: Path) -> None:
    """専門用語集(Glossary)をYAMLまたはJSONファイルに保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = glossary.model_dump(mode="json")

    suffix = path.suffix.lower()
    if suffix == ".json":
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    else:
        text = yaml.safe_dump(
            payload,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
    path.write_text(text, encoding="utf-8")
