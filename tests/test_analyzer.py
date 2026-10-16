from jjargon.analyzer import SudachiAnalyzer
from jjargon.models import Glossary, Term, TermStatus


def test_sudachi_analyzer_extracts_oov_and_compounds() -> None:
    analyzer = SudachiAnalyzer()
    text = (
        "本日の会議では、新システムであるガルoon連携基盤と、"
        "傾斜配分ロジックの改修について議論しました。\n"
        "また、10件の稟議ワークフローを処理しました。"
    )

    candidates = analyzer.analyze(text, include_compounds=True)
    surfaces = {c.surface: c for c in candidates}

    # OOVまたは複合名詞として「ガルoon」「傾斜配分」「稟議ワークフロー」などが抽出されること
    assert any("ガル" in s or "oon" in s for s in surfaces)
    assert "傾斜配分ロジック" in surfaces or "傾斜配分" in surfaces
    assert "稟議ワークフロー" in surfaces

    # 単なる助数詞「10件」はノイズとして除外されていること
    assert "10件" not in surfaces

    # 例文（examples）が保持されていること
    wf_cand = surfaces["稟議ワークフロー"]
    assert wf_cand.is_compound is True
    assert len(wf_cand.examples) == 1
    assert "稟議ワークフロー" in wf_cand.examples[0]


def test_sudachi_analyzer_respects_denied_terms() -> None:
    analyzer = SudachiAnalyzer()
    glossary = Glossary(
        terms=[
            Term(
                canonical_text="稟議ワークフロー",
                status=TermStatus.DENIED,
            )
        ]
    )
    text = "稟議ワークフローを見直しました。"
    candidates = analyzer.analyze(text, glossary=glossary, include_compounds=True)
    surfaces = {c.surface for c in candidates}
    assert "稟議ワークフロー" not in surfaces


def test_sudachi_analyzer_oov_only_mode() -> None:
    analyzer = SudachiAnalyzer()
    text = "顧客管理システムとZXYW123を連携します。"
    candidates = analyzer.analyze(text, include_compounds=False)
    assert all(c.is_oov for c in candidates)
    assert all(not c.is_compound for c in candidates)
