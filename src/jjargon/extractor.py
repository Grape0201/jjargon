import json
import os

from openai import OpenAI

from jjargon.analyzer import SudachiAnalyzer
from jjargon.models import (
    CandidateProposal,
    ExtractionResponse,
    Glossary,
    SudachiCandidate,
    Term,
    TermStatus,
    TextExtractionResponse,
)

SYSTEM_PROMPT = """\
あなたは日本語テキスト（仕様書、Wiki、技術ドキュメント、業務マニュアル等）から専門用語・社内用語（ジャゴン）を抽出・整理する専門AIアシスタントです。
与えられた「入力テキスト」「Sudachi形態素解析による抽出候補」「現在の専門用語集」を照らし合わせ、専門用語集に追加・更新すべき用語候補を出力してください。

## 抽出・判定ルール
1. 新規の専門用語 (proposal_type: "new_term"):
   - 国語辞書に載っていない専門用語、社内用語、システム名、プロジェクト名、略語などを抽出してください。
   - 一般的な日常語や単なる人名・数値、および「却下済み用語リスト」に含まれる語は除外してください。
2. 既存用語への別表記・略称の追加 (proposal_type: "update_existing"):
   - 既存用語集の用語について、新しい類義語・別表記・英語略称がある場合、canonical_text を既存用語と一致させて提案してください。
3. 各フィールドの記述方針:
   - canonical_text: 代表的な正式表記
   - yomi: カタカナ読み
   - aliases: 別表記・略称一覧
   - meaning: 用語の定義・意味
   - term_type: 用語種別（technical_term, system, product, acronym 等）
   - examples: 入力テキスト内の実際の用例抜粋
   - reason: 専門用語と判断した理由
"""


def build_user_prompt(
    text_chunk: str,
    sudachi_candidates: list[SudachiCandidate],
    glossary: Glossary,
    domain: str | None = None,
) -> str:
    sections: list[str] = []
    if domain:
        sections.append(f"## 対象ドメイン\n{domain}")

    active_terms = [
        {
            "canonical_text": t.canonical_text,
            "yomi": t.yomi,
            "aliases": t.aliases,
            "meaning": t.meaning,
            "term_type": t.term_type,
            "domain": t.domain,
        }
        for t in glossary.terms
        if t.status != TermStatus.DENIED
    ]
    if active_terms:
        sections.append(
            "## 現在の専門用語集\n"
            + json.dumps(active_terms, ensure_ascii=False, indent=2)
        )

    denied = {
        form
        for t in glossary.terms
        if t.status == TermStatus.DENIED
        for form in t.all_surface_forms(include_pending=False)
    }
    if denied:
        sections.append(
            f"## 却下済み用語リスト (抽出対象外)\n{', '.join(sorted(denied))}"
        )

    cand_lines = [
        f"- {c.surface} (読み: {c.yomi}, 種別: {'OOV' if c.is_oov else '複合名詞'}, 出現回数: {c.count})"
        for c in sudachi_candidates[:50]
    ]
    sections.append(
        f"## Sudachi抽出候補 (ヒント)\n{chr(10).join(cand_lines) if cand_lines else '（なし）'}"
    )
    sections.append(f"## 入力テキスト\n```text\n{text_chunk}\n```")
    return "\n\n".join(sections)


class JargonExtractor:
    """テキストから専門用語候補を抽出してGlossaryに反映するクラス"""

    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        max_chunk_chars: int = 4000,
    ) -> None:
        self.analyzer = SudachiAnalyzer()
        self.model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self.api_base = api_base or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        self.max_chunk_chars = max_chunk_chars

    def extract_candidates(
        self,
        text_chunk: str,
        sudachi_candidates: list[SudachiCandidate],
        glossary: Glossary,
        domain: str | None = None,
    ) -> ExtractionResponse:
        user_prompt = build_user_prompt(
            text_chunk=text_chunk,
            sudachi_candidates=sudachi_candidates,
            glossary=glossary,
            domain=domain,
        )
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=TextExtractionResponse,
            temperature=0.1,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("LLMからの構造化出力のパースに失敗しました。")
        return parsed.to_extraction_response()

    def chunk_text(self, text: str) -> list[str]:
        sentences = self.analyzer.split_sentences(text)
        if not sentences:
            return []

        chunks: list[str] = []
        current_lines: list[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent) + 1
            if current_lines and current_len + sent_len > self.max_chunk_chars:
                chunks.append("\n".join(current_lines))
                current_lines = [sent]
                current_len = sent_len
            else:
                current_lines.append(sent)
                current_len += sent_len

        if current_lines:
            chunks.append("\n".join(current_lines))

        return chunks

    def extract_and_merge(
        self,
        text: str,
        glossary: Glossary,
        domain: str | None = None,
        include_compounds: bool = True,
    ) -> tuple[list[CandidateProposal], dict[str, list[Term]]]:
        chunks = self.chunk_text(text)
        all_proposals: list[CandidateProposal] = []
        summary: dict[str, list[Term]] = {
            "added": [],
            "updated_pending": [],
            "merged_pending": [],
            "skipped_denied": [],
            "unchanged": [],
        }

        for chunk in chunks:
            sudachi_candidates = self.analyzer.analyze(
                text=chunk,
                glossary=glossary,
                include_compounds=include_compounds,
            )
            response = self.extract_candidates(
                text_chunk=chunk,
                sudachi_candidates=sudachi_candidates,
                glossary=glossary,
                domain=domain,
            )
            for proposal in response.proposals:
                all_proposals.append(proposal)
                term, action = glossary.merge_proposal(proposal, default_domain=domain)
                if (
                    term is not None
                    and action in summary
                    and term not in summary[action]
                ):
                    summary[action].append(term)

        return all_proposals, summary
