from unittest.mock import MagicMock

import pytest
from rich.console import Console

from jjargon.extractor import JargonExtractor, build_user_prompt
from jjargon.models import (
    CandidateProposal,
    ExtractionResponse,
    Glossary,
    SudachiCandidate,
    Term,
    TermStatus,
    TextExtractionProposal,
    TextExtractionResponse,
)
from jjargon.reviewer import interactive_review


def test_jargon_extractor_pipeline_and_interactive_review(monkeypatch) -> None:
    glossary = Glossary(
        terms=[
            Term(
                canonical_text="Kintone",
                yomi="キントーン",
                status=TermStatus.APPROVED,
            )
        ]
    )

    mock_response = ExtractionResponse(
        proposals=[
            CandidateProposal(
                proposal_type="new_term",
                canonical_text="傾斜配分マトリクス",
                yomi="ケイシャハイブンマトリクス",
                aliases=["傾斜マトリクス"],
                meaning="予算を特定比率で配分するための算出表",
                term_type="business_term",
                examples=["来期予算は傾斜配分マトリクスを用いて算出します。"],
            ),
            CandidateProposal(
                proposal_type="update_existing",
                canonical_text="Kintone",
                recognized_texts=["筋斗雲"],
                examples=["申請は筋斗雲から提出してください。"],
                reason="キントーンの音声認識誤変換",
            ),
            CandidateProposal(
                proposal_type="new_term",
                canonical_text="不要候補テスト",
                yomi="フヨウコウホテスト",
                meaning="却下テスト用",
            ),
        ]
    )

    extractor = JargonExtractor()
    monkeypatch.setattr(
        extractor, "extract_candidates", lambda *args, **kwargs: mock_response
    )

    sample_text = (
        "来期予算は傾斜配分マトリクスを用いて算出します。"
        "申請は筋斗雲から提出してください。"
    )
    proposals, summary = extractor.extract_and_merge(
        text=sample_text,
        glossary=glossary,
        domain="経営企画部",
    )

    assert len(proposals) == 3
    assert len(summary["added"]) == 2
    assert len(summary["updated_pending"]) == 1

    # レビュー待ちは 3件（既存Kintoneのpending_updates 1件 ＋ 新規pending 2件）
    pending_list = glossary.get_pending_reviews()
    assert len(pending_list) == 3

    # 対話レビューのシミュレーション:
    # 1件目 (Kintoneの追記): 'a' (承認)
    # 2件目 (傾斜配分マトリクス): 'e' (編集して承認)
    # 3件目 (不要候補テスト): 'd' (却下)
    responses = iter(
        [
            "a",  # 1件目: Kintoneのpending_updatesを承認
            "e",  # 2件目: 傾斜配分マトリクスを編集
            "傾斜配分マトリクス",  # canonical_text
            "ケイシャハイブンマトリクス",  # yomi
            "傾斜マトリクス, 配分表",  # aliases
            "",  # recognized_texts
            "予算を特定比率で配分するための算出マトリクス",  # meaning
            "business_term",  # term_type
            "経営企画部",  # domain
            "d",  # 3件目: 不要候補テストを却下
        ]
    )

    stats = interactive_review(
        glossary=glossary,
        console=Console(quiet=True),
        input_fn=lambda _prompt, _default="": next(responses),
    )

    assert stats["approved"] == 2
    assert stats["denied"] == 1
    assert stats["skipped"] == 0

    # 承認後の検証
    kintone_term = glossary.find_by_canonical("Kintone")
    assert kintone_term is not None
    assert "筋斗雲" in kintone_term.recognized_texts
    assert kintone_term.pending_updates is None

    matrix_term = glossary.find_by_canonical("傾斜配分マトリクス")
    assert matrix_term is not None
    assert matrix_term.status == TermStatus.APPROVED
    assert matrix_term.aliases == ["傾斜マトリクス", "配分表"]

    # 却下された「不要候補テスト」が denied になっていること
    denied_term = glossary.find_by_canonical("不要候補テスト")
    assert denied_term is not None
    assert denied_term.status == TermStatus.DENIED


def test_jargon_extractor_extract_candidates(monkeypatch) -> None:
    extractor = JargonExtractor(model="gpt-4o-mini", api_key="test-key")
    mock_llm_response = TextExtractionResponse(
        proposals=[
            TextExtractionProposal(
                proposal_type="new_term",
                canonical_text="テスト用語",
                yomi="テストヨウゴ",
            )
        ]
    )
    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_llm_response
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_parse = MagicMock(return_value=mock_completion)
    monkeypatch.setattr(extractor.client.beta.chat.completions, "parse", mock_parse)

    result = extractor.extract_candidates(
        text_chunk="これはテスト用語の文章です。",
        sudachi_candidates=[],
        glossary=Glossary(),
    )
    assert len(result.proposals) == 1
    assert result.proposals[0].canonical_text == "テスト用語"
    assert result.proposals[0].recognized_texts == []
    mock_parse.assert_called_once()
    assert mock_parse.call_args.kwargs["response_format"] == TextExtractionResponse


def test_jargon_extractor_extract_candidates_raises_on_none(monkeypatch) -> None:
    extractor = JargonExtractor(model="gpt-4o-mini", api_key="test-key")
    mock_choice = MagicMock()
    mock_choice.message.parsed = None
    mock_choice.message.content = "invalid response"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    monkeypatch.setattr(
        extractor.client.beta.chat.completions,
        "parse",
        MagicMock(return_value=mock_completion),
    )

    with pytest.raises(ValueError, match="LLMからの構造化出力のパースに失敗しました"):
        extractor.extract_candidates(
            text_chunk="テスト",
            sudachi_candidates=[],
            glossary=Glossary(),
        )


def test_jargon_extractor_defaults(monkeypatch) -> None:
    for var in ["OPENAI_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY"]:
        monkeypatch.delenv(var, raising=False)

    extractor = JargonExtractor()
    assert extractor.model == "gpt-4o-mini"
    assert extractor.api_base is None
    assert extractor.api_key == "not-needed"


def test_jargon_extractor_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret-key")
    monkeypatch.setenv("OPENAI_MODEL", "my-custom-model")

    extractor = JargonExtractor()
    assert extractor.model == "my-custom-model"
    assert extractor.api_base == "https://api.example.com/v1"
    assert extractor.api_key == "env-secret-key"


def test_jargon_extractor_args_override_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.env.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")

    extractor = JargonExtractor(
        model="gpt-4o",
        api_base="https://api.override.com",
        api_key="arg-key",
    )
    assert extractor.model == "gpt-4o"
    assert extractor.api_base == "https://api.override.com"
    assert extractor.api_key == "arg-key"


def test_build_user_prompt_filters_irrelevant_glossary_terms() -> None:
    glossary = Glossary(
        terms=[
            # 1. 表層形が直接出現する用語
            Term(
                canonical_text="Kintone",
                yomi="キントーン",
                meaning="業務アプリ構築サービス",
                status=TermStatus.APPROVED,
            ),
            # 2. 読みがSudachi候補と一致する用語（同音異義語・別表記の検出用）
            Term(
                canonical_text="高速炉",
                yomi="コウソクロ",
                status=TermStatus.APPROVED,
            ),
            # 3. 構成語が部分一致する用語（略称の検出用）
            Term(
                canonical_text="傾斜配分マトリクス",
                yomi="ケイシャハイブンマトリクス",
                status=TermStatus.APPROVED,
            ),
            # 4. チャンクに全く関係ない用語（除外されるべき）
            Term(
                canonical_text="無関係プロジェクトX",
                yomi="ムカンケイプロジェクトエックス",
                status=TermStatus.APPROVED,
            ),
            # 5. チャンク内に出現する却下済み用語（プロンプトに含まれるべき）
            Term(
                canonical_text="スケジュール",
                status=TermStatus.DENIED,
            ),
            # 6. チャンク内に出現しない却下済み用語（除外されるべき）
            Term(
                canonical_text="不要な却下用語Z",
                status=TermStatus.DENIED,
            ),
        ]
    )

    text_chunk = (
        "Kintoneでスケジュールを確認し、拘束炉と傾斜マトリクスの資料を作成する。"
    )
    candidates = [
        SudachiCandidate(
            surface="拘束炉", yomi="コウソクロ", is_oov=False, is_compound=True
        ),
        SudachiCandidate(
            surface="傾斜マトリクス",
            yomi="ケイシャマトリクス",
            is_oov=False,
            is_compound=True,
        ),
    ]

    prompt = build_user_prompt(
        text_chunk=text_chunk,
        sudachi_candidates=candidates,
        glossary=glossary,
    )

    assert "Kintone" in prompt
    assert "高速炉" in prompt
    assert "傾斜配分マトリクス" in prompt
    assert "無関係プロジェクトX" not in prompt

    assert "スケジュール" in prompt
    assert "不要な却下用語Z" not in prompt


def test_build_user_prompt_respects_max_glossary_terms() -> None:
    terms = [
        Term(
            canonical_text=f"テスト用語{i}",
            yomi=f"テストヨウゴ{i}",
            status=TermStatus.APPROVED,
        )
        for i in range(10)
    ]
    glossary = Glossary(terms=terms)
    text_chunk = " ".join(t.canonical_text for t in terms)

    prompt = build_user_prompt(
        text_chunk=text_chunk,
        sudachi_candidates=[],
        glossary=glossary,
        max_glossary_terms=3,
    )

    included_count = sum(1 for t in terms if f'"{t.canonical_text}"' in prompt)
    assert included_count == 3


def test_extract_and_merge_on_progress_callback(monkeypatch) -> None:
    extractor = JargonExtractor(max_chunk_chars=20)
    monkeypatch.setattr(
        extractor,
        "extract_candidates",
        lambda *args, **kwargs: ExtractionResponse(proposals=[]),
    )

    progress_steps: list[tuple[int, int]] = []
    text = "一文目のテキストです。\n二文目のテキストです。\n三文目のテキストです。"
    extractor.extract_and_merge(
        text=text,
        glossary=Glossary(),
        on_progress=lambda current, total: progress_steps.append((current, total)),
    )

    assert progress_steps == [(0, 3), (1, 3), (2, 3), (3, 3)]

