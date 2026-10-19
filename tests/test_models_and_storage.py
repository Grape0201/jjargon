from pathlib import Path

from jjargon.models import CandidateProposal, Glossary, Term, TermStatus
from jjargon.storage import load_glossary, save_glossary


def test_glossary_merge_new_and_existing_approved_updates() -> None:
    glossary = Glossary(
        terms=[
            Term(
                id="term-kintone",
                canonical_text="Kintone",
                yomi="キントーン",
                aliases=["kintone"],
                recognized_texts=["均等ん"],
                meaning="業務アプリ構築クラウドサービス",
                term_type="service",
                domain="情シス",
                status=TermStatus.APPROVED,
            ),
            Term(
                id="term-denied",
                canonical_text="スケジュール",
                status=TermStatus.DENIED,
            ),
        ]
    )

    # 1. 新規用語提案のマージ
    p_new = CandidateProposal(
        proposal_type="new_term",
        canonical_text="SmartHR",
        yomi="スマートエイチアール",
        aliases=["スマHR"],
        recognized_texts=["スマートエッチアール"],
        meaning="労務管理クラウド",
        term_type="service",
        examples=["スマートエッチアールで年末調整を行う"],
    )
    term_new, action_new = glossary.merge_proposal(p_new, default_domain="人事部")
    assert action_new == "added"
    assert term_new is not None
    assert term_new.status == TermStatus.PENDING
    assert term_new.domain == "人事部"
    assert "スマートエッチアール" in term_new.recognized_texts

    # 2. 既存の承認済み用語（Kintone）に対する新しい誤認識・別表記提案
    p_update = CandidateProposal(
        proposal_type="update_existing",
        canonical_text="Kintone",
        aliases=["キントーンアプリ"],
        recognized_texts=["筋斗雲", "均等ん"],  # 「均等ん」は既存、「筋斗雲」は新規
        examples=["筋斗雲のスペースに投稿しました"],
        reason="音声認識の同音異義語誤変換",
    )
    term_upd, action_upd = glossary.merge_proposal(p_update)
    assert action_upd == "updated_pending"
    assert term_upd is not None
    assert term_upd.status == TermStatus.APPROVED
    assert term_upd.pending_updates is not None
    assert term_upd.pending_updates.aliases == ["キントーンアプリ"]
    assert term_upd.pending_updates.recognized_texts == ["筋斗雲"]
    assert term_upd.has_pending_review() is True

    # 差分を承認適用すると本体にマージされ pending_updates がクリアされること
    term_upd.apply_pending_updates()
    assert term_upd.pending_updates is None
    assert "キントーンアプリ" in term_upd.aliases
    assert "筋斗雲" in term_upd.recognized_texts
    assert "筋斗雲のスペースに投稿しました" in term_upd.examples

    # 3. 却下済み用語（denied）はスキップされること
    p_denied = CandidateProposal(
        proposal_type="new_term",
        canonical_text="スケジュール",
    )
    _, action_denied = glossary.merge_proposal(p_denied)
    assert action_denied == "skipped_denied"


def test_storage_yaml_and_json_roundtrip(tmp_path: Path) -> None:
    glossary = Glossary(
        terms=[
            Term(
                canonical_text="稟議システム",
                yomi="リンギシステム",
                aliases=["電子稟議"],
                recognized_texts=["林檎システム"],
                meaning="社内決裁ワークフロー",
                term_type="system",
                domain="総務部",
                status=TermStatus.APPROVED,
            )
        ]
    )

    yaml_path = tmp_path / "glossary.yaml"
    json_path = tmp_path / "glossary.json"

    save_glossary(glossary, yaml_path)
    save_glossary(glossary, json_path)

    loaded_yaml = load_glossary(yaml_path)
    loaded_json = load_glossary(json_path)

    assert len(loaded_yaml.terms) == 1
    assert loaded_yaml.terms[0].canonical_text == "稟議システム"
    assert loaded_yaml.terms[0].recognized_texts == ["林檎システム"]

    assert len(loaded_json.terms) == 1
    assert loaded_json.terms[0].canonical_text == "稟議システム"


def test_glossary_merge_fallback_when_canonical_is_misrecognition() -> None:
    glossary = Glossary(
        terms=[
            Term(
                id="term-fast-reactor",
                canonical_text="高速炉",
                yomi="コウソクロ",
                status=TermStatus.APPROVED,
            )
        ]
    )

    # LLMが誤って誤変換「拘束炉」を canonical_text にし、aliases に「高速炉」を入れた場合
    proposal = CandidateProposal(
        proposal_type="new_term",
        canonical_text="拘束炉",
        yomi="コウソクロ",
        aliases=["高速炉"],
        recognized_texts=["拘束路"],
        meaning="高速炉（Fast Reactor）の誤認識と思われる表記",
        reason="音声認識による同音異義語の誤変換",
    )

    term, action = glossary.merge_proposal(proposal)
    assert term is not None
    assert term.canonical_text == "高速炉"
    assert action == "updated_pending"
    assert term.pending_updates is not None
    # 「拘束炉」「拘束路」の双方が誤認識として pending_updates に追加されていること
    assert "拘束炉" in term.pending_updates.recognized_texts
    assert "拘束路" in term.pending_updates.recognized_texts


def test_glossary_merge_normalized_deduplication() -> None:
    """プロンプトから除外された既存用語と大文字小文字・全角半角揺れがある提案でも重複登録されないこと"""
    glossary = Glossary(
        terms=[
            Term(
                id="term-kintone",
                canonical_text="Kintone",
                yomi="キントーン",
                aliases=["サイボウズKintone"],
                status=TermStatus.APPROVED,
            ),
            Term(
                id="term-denied",
                canonical_text="GitHub",
                status=TermStatus.DENIED,
            ),
        ]
    )

    # 1. 全角・小文字の「ｋｉｎｔｏｎｅ」が new_term として提案された場合、新規追加にならず既存Kintoneに紐づくこと
    p_dup = CandidateProposal(
        proposal_type="new_term",
        canonical_text="ｋｉｎｔｏｎｅ",
        yomi="キントーン",
    )
    term, action = glossary.merge_proposal(p_dup)
    assert term is not None
    assert term.id == "term-kintone"
    assert action == "unchanged"
    assert len(glossary.terms) == 2

    # 2. 新しい別表記を伴う場合は updated_pending になること
    p_with_new_alias = CandidateProposal(
        proposal_type="new_term",
        canonical_text="kintone",
        aliases=["キントーン基盤"],
    )
    term2, action2 = glossary.merge_proposal(p_with_new_alias)
    assert term2 is not None
    assert term2.id == "term-kintone"
    assert action2 == "updated_pending"
    assert term2.pending_updates is not None
    assert term2.pending_updates.aliases == ["キントーン基盤"]

    # 3. 却下済み用語（GitHub）が小文字・全角混じりで再提案されても skipped_denied になること
    p_denied_norm = CandidateProposal(
        proposal_type="new_term",
        canonical_text="ｇｉｔｈｕｂ",
    )
    _, action_denied = glossary.merge_proposal(p_denied_norm)
    assert action_denied == "skipped_denied"
