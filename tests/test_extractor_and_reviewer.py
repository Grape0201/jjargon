from unittest.mock import MagicMock

import pytest
from rich.console import Console

from jjargon.extractor import (
    ExtractionMode,
    JargonExtractor,
    LLMConfig,
    SystemOneClient,
    SystemOneConfig,
    build_user_prompt,
)
from jjargon.models import (
    CandidateProposal,
    CompactExtractionProposal,
    CompactExtractionResponse,
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
    extractor = JargonExtractor(
        llm_config=LLMConfig(model="gpt-4o-mini", api_key="test-key")
    )
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
    extractor = JargonExtractor(
        llm_config=LLMConfig(model="gpt-4o-mini", api_key="test-key")
    )
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


def test_llm_config_defaults(monkeypatch) -> None:
    for var in ["OPENAI_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY"]:
        monkeypatch.delenv(var, raising=False)

    config = LLMConfig.from_env()
    assert config.model == "gpt-4o-mini"
    assert config.api_base is None
    assert config.base is None
    assert config.api_key == "not-needed"
    assert config.key == "not-needed"


def test_llm_config_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret-key")
    monkeypatch.setenv("OPENAI_MODEL", "my-custom-model")

    config = LLMConfig.from_env()
    assert config.model == "my-custom-model"
    assert config.api_base == "https://api.example.com/v1"
    assert config.base == "https://api.example.com/v1"
    assert config.api_key == "env-secret-key"
    assert config.key == "env-secret-key"


def test_llm_config_explicit_base_key_model() -> None:
    config = LLMConfig(
        model="gpt-4o",
        base="https://api.override.com",
        key="arg-key",
    )
    assert config.model == "gpt-4o"
    assert config.api_base == "https://api.override.com"
    assert config.base == "https://api.override.com"
    assert config.api_key == "arg-key"
    assert config.key == "arg-key"


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
    extractor = JargonExtractor()
    extractor.max_chunk_chars = 20
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


def test_compact_mode_enriches_yomi_and_examples_from_sudachi(monkeypatch) -> None:
    extractor = JargonExtractor(
        llm_config=LLMConfig(model="gpt-4o-mini", api_key="test-key", compact_mode=True)
    )
    mock_llm_response = CompactExtractionResponse(
        proposals=[
            CompactExtractionProposal(
                proposal_type="new_term",
                canonical_text="傾斜配分マトリクス",
                aliases=["傾斜マトリクス"],
                term_type="business_term",
                meaning="予算を特定比率で配分する算出表",
            )
        ]
    )
    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_llm_response
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_parse = MagicMock(return_value=mock_completion)
    monkeypatch.setattr(extractor.client.beta.chat.completions, "parse", mock_parse)

    candidates = [
        SudachiCandidate(
            surface="傾斜配分マトリクス",
            normalized_form="傾斜配分マトリクス",
            yomi="ケイシャハイブンマトリクス",
            is_oov=False,
            is_compound=True,
            examples=["来期予算は傾斜配分マトリクスを用いて算出します。"],
        )
    ]

    result = extractor.extract_candidates(
        text_chunk="来期予算は傾斜配分マトリクスを用いて算出します。",
        sudachi_candidates=candidates,
        glossary=Glossary(),
        domain="経営企画部",
    )

    mock_parse.assert_called_once()
    assert mock_parse.call_args.kwargs["response_format"] == CompactExtractionResponse
    assert len(result.proposals) == 1
    p = result.proposals[0]
    assert p.canonical_text == "傾斜配分マトリクス"
    assert p.yomi == "ケイシャハイブンマトリクス"
    assert p.examples == ["来期予算は傾斜配分マトリクスを用いて算出します。"]
    assert p.domain == "経営企画部"


def test_systemone_hybrid_filter_and_chunk_skip(monkeypatch) -> None:
    s1_config = SystemOneConfig(
        model="kev:latest", url="http://localhost:11434/v1/systemone", threshold=0.5
    )

    extractor = JargonExtractor(
        mode=ExtractionMode.HYBRID,
        llm_config=LLMConfig(compact_mode=True),
        system_one_config=s1_config,
    )
    extractor.max_chunk_chars = 30
    assert extractor.system_one is not None

    def fake_s1_call(state: dict, questions: dict) -> dict:
        assert extractor.system_one is not None
        extractor.system_one.call_count += 1
        cands = state.get("candidates", {})
        answers = {}
        for cid, cinfo in cands.items():
            # 「傾斜配分マトリクス」のみ高スコア、一般語ノイズは低スコア
            score = 0.88 if "傾斜配分" in cinfo.get("surface", "") else 0.12
            answers[f"{cid}_is_jargon"] = {"type": "noul", "noul": score}
        return answers

    monkeypatch.setattr(extractor.system_one, "call", fake_s1_call)

    llm_called_chunks: list[str] = []

    def fake_extract_candidates(
        text_chunk: str,
        sudachi_candidates: list[SudachiCandidate],
        glossary: Glossary,
        domain: str | None = None,
    ) -> ExtractionResponse:
        llm_called_chunks.append(text_chunk)
        return ExtractionResponse(
            proposals=[
                CandidateProposal(
                    proposal_type="new_term",
                    canonical_text=c.surface,
                    meaning="テスト定義",
                )
                for c in sudachi_candidates
            ]
        )

    monkeypatch.setattr(extractor, "extract_candidates", fake_extract_candidates)

    # チャンク1は専門用語あり、チャンク2は一般語（「確認作業」）のみ
    text = (
        "来期予算は傾斜配分マトリクスで算出します。\n画面での確認作業を完了しました。"
    )
    proposals, summary = extractor.extract_and_merge(
        text=text,
        glossary=Glossary(),
        domain="経営企画部",
    )

    assert extractor.last_stats["total_chunks"] == 2
    assert extractor.last_stats["skipped_chunks"] == 1
    assert extractor.last_stats["llm_calls"] == 1
    assert len(llm_called_chunks) == 1
    assert len(proposals) == 1
    assert proposals[0].canonical_text == "傾斜配分マトリクス"
    assert proposals[0].yomi == "ケイシャハイブンマトリクス"
    assert len(summary["added"]) == 1


def test_systemone_only_mode_extracts_and_merges_without_llm(monkeypatch) -> None:
    s1_config = SystemOneConfig(
        model="kev:latest", url="http://localhost:11434/v1/systemone", threshold=0.5
    )
    extractor = JargonExtractor(
        mode=ExtractionMode.SYSTEM_ONE_ONLY,
        system_one_config=s1_config,
    )
    assert extractor.system_one is not None

    def fake_s1_call(state: dict, questions: dict) -> dict:
        assert extractor.system_one is not None
        extractor.system_one.call_count += 1
        cands = state.get("candidates", {})
        answers = {}
        for qkey in questions:
            cid = qkey.split("_")[0]
            surf = cands.get(cid, {}).get("surface", "")
            if qkey.endswith("_is_jargon"):
                score = 0.85 if surf in {"傾斜マトリクス", "SmartHR"} else 0.10
                answers[qkey] = {"type": "noul", "noul": score}
            elif qkey.endswith("_term_type"):
                t_type = "product" if surf == "SmartHR" else "business_term"
                answers[qkey] = {"type": "choice", "choice": t_type}
            elif qkey.endswith("_existing_match"):
                # 「傾斜マトリクス」は既存用語 term_0（傾斜配分マトリクス）にマッチ、「SmartHR」は新規
                choice = "term_0" if surf == "傾斜マトリクス" else "new_term"
                answers[qkey] = {"type": "choice", "choice": choice}
        return answers

    monkeypatch.setattr(extractor.system_one, "call", fake_s1_call)

    glossary = Glossary(
        terms=[
            Term(
                canonical_text="傾斜配分マトリクス",
                yomi="ケイシャハイブンマトリクス",
                status=TermStatus.APPROVED,
            )
        ]
    )

    text = "来期予算は傾斜マトリクスで算出し、年末調整はSmartHRで行います。"
    proposals, summary = extractor.extract_and_merge(
        text=text,
        glossary=glossary,
        domain="人事企画",
    )

    # LLM呼び出しが0回で、新規1件(SmartHR)＋既存更新1件(傾斜配分マトリクスの別表記として傾斜マトリクス)が抽出されること
    assert extractor.last_stats["llm_calls"] == 0
    assert extractor.last_stats["systemone_calls"] >= 2
    assert len(proposals) == 2
    assert len(summary["added"]) == 1
    assert summary["added"][0].canonical_text == "SmartHR"
    assert summary["added"][0].term_type == "product"
    assert len(summary["updated_pending"]) == 1
    existing_term = glossary.find_by_canonical("傾斜配分マトリクス")
    assert existing_term is not None
    assert existing_term.pending_updates is not None
    assert "傾斜マトリクス" in existing_term.pending_updates.aliases


def test_systemone_client_missing_config_raises_error(monkeypatch) -> None:
    monkeypatch.delenv("SYSTEMONE_MODEL", raising=False)
    monkeypatch.delenv("SYSTEMONE_URL", raising=False)

    with pytest.raises(ValueError, match="System One モデルが指定されていません"):
        SystemOneClient(model=None, url="http://localhost:11434/v1/systemone")

    with pytest.raises(
        ValueError, match="System One エンドポイントURLが指定されていません"
    ):
        SystemOneClient(model="kev:latest", url=None)

    with pytest.raises(ValueError, match="System One モデルが指定されていません"):
        JargonExtractor(mode=ExtractionMode.SYSTEM_ONE_ONLY)


def test_system_one_config_struct() -> None:
    config = SystemOneConfig(
        model="kev:v2",
        base_url="http://localhost:11434/v1/systemone",
        threshold=0.8,
    )
    assert config.model == "kev:v2"
    assert config.url == "http://localhost:11434/v1/systemone"
    assert config.base_url == "http://localhost:11434/v1/systemone"
    assert config.threshold == 0.8


def test_extract_and_merge_mode_override(monkeypatch) -> None:
    s1_config = SystemOneConfig(
        model="kev:latest", url="http://localhost:11434/v1/systemone", threshold=0.5
    )
    llm_config = LLMConfig(model="gpt-4o-mini", api_key="test-key")

    # extractor は HYBRID で初期化
    extractor = JargonExtractor(
        mode=ExtractionMode.HYBRID,
        llm_config=llm_config,
        system_one_config=s1_config,
    )

    # extract_and_merge 呼び出し時に SYSTEM_TWO_ONLY を指定してオーバーライド
    monkeypatch.setattr(
        extractor,
        "extract_candidates",
        lambda *args, **kwargs: ExtractionResponse(proposals=[]),
    )

    _proposals, _summary = extractor.extract_and_merge(
        text="テストテキストです。",
        glossary=Glossary(),
        mode=ExtractionMode.SYSTEM_TWO_ONLY,
    )
    assert extractor.last_stats["llm_calls"] == 1
    assert extractor.last_stats["systemone_calls"] == 0


def test_jargon_extractor_symmetric_components() -> None:
    s1_config = SystemOneConfig(
        model="kev:latest", url="http://localhost:11434/v1/systemone", threshold=0.6
    )
    llm_config = LLMConfig(model="gpt-4o-mini", api_key="test-key", compact_mode=True)

    # 1. HYBRID モード初期化: 両クライアントが初期化されること
    extractor_hybrid = JargonExtractor(
        mode=ExtractionMode.HYBRID,
        llm_config=llm_config,
        system_one_config=s1_config,
    )
    assert extractor_hybrid.mode == ExtractionMode.HYBRID
    assert extractor_hybrid.system_one is not None
    assert extractor_hybrid.system_two is not None
    assert extractor_hybrid.compact_mode is True

    # 2. SYSTEM_ONE_ONLY モード: System One のみ初期化されること
    extractor_s1 = JargonExtractor(
        mode=ExtractionMode.SYSTEM_ONE_ONLY,
        system_one_config=s1_config,
    )
    assert extractor_s1.mode == ExtractionMode.SYSTEM_ONE_ONLY
    assert extractor_s1.system_one is not None
    assert extractor_s1.system_two is None

    # 3. SYSTEM_TWO_ONLY モード: System Two のみ初期化されること
    extractor_s2 = JargonExtractor(
        mode=ExtractionMode.SYSTEM_TWO_ONLY,
        llm_config=llm_config,
    )
    assert extractor_s2.mode == ExtractionMode.SYSTEM_TWO_ONLY
    assert extractor_s2.system_one is None
    assert extractor_s2.system_two is not None
