import csv
import io
import re
from pathlib import Path

from jjargon.models import Glossary, TermStatus

KATAKANA_ONLY_RE = re.compile(r"^[ァ-ヴー・]+$")


def _hiragana_to_katakana(text: str) -> str:
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ん" else ch for ch in text)


def _normalize_reading(yomi: str, fallback_surface: str) -> str:
    """Sudachi辞書用に読みをカタカナに正規化する"""
    candidate = _hiragana_to_katakana(yomi.strip())
    if candidate and KATAKANA_ONLY_RE.match(candidate):
        return candidate
    fallback = _hiragana_to_katakana(fallback_surface.strip())
    if fallback and KATAKANA_ONLY_RE.match(fallback):
        return fallback
    return candidate or fallback_surface.strip()


def generate_sudachi_csv(
    glossary: Glossary,
    include_aliases: bool = True,
    cost: int = 5000,
) -> str:
    """
    status == 'approved' の専門用語からSudachiユーザー辞書形式（18カラムCSV）を生成する。
    aliases も登録する場合、正規化表記（13列目）に canonical_text を設定することで表記揺れ正規化に活用できる。
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    seen_surfaces: set[str] = set()

    for term in glossary.terms:
        if term.status != TermStatus.APPROVED:
            continue

        surfaces = [term.canonical_text]
        if include_aliases:
            surfaces.extend(term.aliases)

        for surface in surfaces:
            surf = surface.strip()
            if not surf or surf in seen_surfaces:
                continue
            seen_surfaces.add(surf)

            reading = _normalize_reading(term.yomi, surf)
            row = [
                surf,  # 0: 見出し
                4785,  # 1: 左連接ID (名詞-固有名詞-一般の標準値)
                4785,  # 2: 右連接ID
                cost,  # 3: コスト
                surf,  # 4: 見出し（単語表記）
                "名詞",  # 5: 品詞1
                "固有名詞",  # 6: 品詞2
                "一般",  # 7: 品詞3
                "*",  # 8: 品詞4
                "*",  # 9: 品詞5
                "*",  # 10: 品詞6
                reading,  # 11: 読み（カタカナ）
                term.canonical_text,  # 12: 正規化表記（canonical_textに統一）
                "*",  # 13: 辞書形ID
                "A",  # 14: 分割タイプ
                "*",  # 15: A単位分割情報
                "*",  # 16: B単位分割情報
                "*",  # 17: 未使用
            ]
            writer.writerow(row)

    return output.getvalue()


def export_sudachi_csv(
    glossary: Glossary,
    output_path: Path,
    include_aliases: bool = True,
    cost: int = 5000,
) -> int:
    """Sudachiユーザー辞書CSVをファイルに書き出し、出力行数を返す"""
    csv_text = generate_sudachi_csv(
        glossary=glossary,
        include_aliases=include_aliases,
        cost=cost,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(csv_text, encoding="utf-8")
    return len([line for line in csv_text.splitlines() if line.strip()])
