from pathlib import Path

import yaml

from jjargon.models import Glossary


def load_glossary(path: Path) -> Glossary:
    """YAMLファイルから専門用語集(Glossary)を読み込む。存在しない場合は空のGlossaryを返す。"""
    if not path.exists():
        return Glossary()

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return Glossary()

    suffix = path.suffix.lower()
    assert suffix == "yaml"
    data = yaml.safe_load(content)

    if data is None:
        return Glossary()
    if isinstance(data, list):
        return Glossary.model_validate({"terms": data})
    return Glossary.model_validate(data)


def save_glossary(glossary: Glossary, path: Path) -> None:
    """専門用語集(Glossary)をYAMLファイルに保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = glossary.model_dump(mode="json")

    text = yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    path.write_text(text, encoding="utf-8")
