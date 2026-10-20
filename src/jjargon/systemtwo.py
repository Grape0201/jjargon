from __future__ import annotations

import json

from openai import BadRequestError, OpenAI

from jjargon.models import (
    CompactExtractionResponse,
    ExtractionResponse,
    Glossary,
    LLMConfig,
    SudachiCandidate,
    Term,
    TermStatus,
    TextExtractionResponse,
    enrich_proposals_from_sudachi,
    normalize_key,
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

COMPACT_SYSTEM_PROMPT = """\
あなたは日本語テキストから専門用語・社内用語を抽出・整理するAIアシスタントです。
与えられた「入力テキスト」「Sudachi形態素解析による抽出候補」「現在の専門用語集」を照らし合わせ、専門用語集に追加・更新すべき用語候補のみを最小限のフィールドで出力してください。
（読み・用例・理由はシステム側でSudachi解析結果から自動補完するため出力不要です）

## 抽出・判定ルール
1. 新規の専門用語 (proposal_type: "new_term"):
   - 一般辞書に載っていない専門用語、社内用語、システム名、製品名、プロジェクト名、略語などを抽出してください。
   - 一般的な日常語、単なる一般名詞の連結、人名、および「却下済み用語リスト」の語は除外してください。
2. 既存用語への別表記・略称の追加 (proposal_type: "update_existing"):
   - 既存用語集の用語に対する新しい別表記・略称がある場合、canonical_text を既存用語と一致させ、aliases に追加表記を入れて提案してください（meaning は空文字で構いません）。
3. 各フィールドの記述方針:
   - canonical_text: 代表的な正式表記
   - aliases: 別表記・略称一覧
   - term_type: technical_term, business_term, system, product, acronym, organization, workflow のいずれか
   - meaning: 1文以内の簡潔な用語の意味・定義
"""


def _has_common_subword(text_a: str, text_b: str, min_len: int = 2) -> bool:
    """2つの正規化済み文字列間に min_len 文字以上の共通部分文字列（構成語の手がかり）があるか判定する"""
    if len(text_a) < min_len or len(text_b) < min_len:
        return False
    shorter, longer = (
        (text_a, text_b) if len(text_a) <= len(text_b) else (text_b, text_a)
    )
    return any(
        shorter[i : i + min_len] in longer for i in range(len(shorter) - min_len + 1)
    )


def select_relevant_terms(
    text_chunk: str,
    sudachi_candidates: list[SudachiCandidate],
    glossary: Glossary,
    max_terms: int = 50,
) -> list[Term]:
    """
    チャンク本文およびSudachi抽出候補に関連する有効な専門用語（status != DENIED）のみを抽出し、
    関連度スコア順で最大 max_terms 件まで返す。
    """
    if max_terms <= 0:
        return []

    norm_chunk = normalize_key(text_chunk)
    cand_surfaces = {
        normalize_key(form)
        for c in sudachi_candidates
        for form in (c.surface, c.normalized_form)
        if form and normalize_key(form)
    }
    cand_yomis = {
        normalize_key(c.yomi)
        for c in sudachi_candidates
        if c.yomi and len(normalize_key(c.yomi)) >= 2
    }
    long_cand_surfaces = [s for s in cand_surfaces if len(s) >= 2]
    long_cand_yomis = [y for y in cand_yomis if len(y) >= 3]

    scored_terms: list[tuple[int, int, int, Term]] = []

    for idx, term in enumerate(glossary.terms):
        if term.status == TermStatus.DENIED:
            continue

        norm_forms = [
            normalize_key(f)
            for f in term.all_surface_forms(include_pending=True)
            if f and normalize_key(f)
        ]
        norm_yomi = normalize_key(term.yomi) if term.yomi else ""

        score = 0
        # Score 3: 表層形（正式表記・別表記・誤認識）がチャンク本文またはSudachi候補に出現
        if any(nf in norm_chunk or nf in cand_surfaces for nf in norm_forms):
            score = 3
        else:
            # Score 2: 読みの一致、または構成語（2文字以上）の部分一致（略称・別表記の検出用）
            yomi_match = bool(
                len(norm_yomi) >= 3
                and (
                    norm_yomi in norm_chunk
                    or norm_yomi in cand_yomis
                    or any(norm_yomi in cy or cy in norm_yomi for cy in long_cand_yomis)
                )
            )
            subword_match = any(
                _has_common_subword(cs, nf, min_len=2)
                for nf in norm_forms
                for cs in long_cand_surfaces
            )
            if yomi_match or subword_match:
                score = 2

        if score > 0:
            scored_terms.append((score, len(term.canonical_text), -idx, term))

    scored_terms.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [item[3] for item in scored_terms[:max_terms]]


def select_relevant_denied_surfaces(
    text_chunk: str,
    sudachi_candidates: list[SudachiCandidate],
    glossary: Glossary,
) -> list[str]:
    """
    却下済み用語（status == DENIED）のうち、チャンク本文またはSudachi候補に出現する表層形のみを返す。
    """
    norm_chunk = normalize_key(text_chunk)
    cand_surfaces = {
        normalize_key(form)
        for c in sudachi_candidates
        for form in (c.surface, c.normalized_form)
        if form and normalize_key(form)
    }

    matched: set[str] = set()
    for term in glossary.terms:
        if term.status != TermStatus.DENIED:
            continue
        for form in term.all_surface_forms(include_pending=False):
            nf = normalize_key(form)
            if nf and (nf in norm_chunk or nf in cand_surfaces):
                matched.add(form)
    return sorted(matched)


def _format_term_for_prompt(term: Term) -> dict[str, object]:
    """プロンプト埋め込み用に空フィールドを省いた辞書を作成する"""
    entry: dict[str, object] = {"canonical_text": term.canonical_text}
    if term.yomi:
        entry["yomi"] = term.yomi
    if term.aliases:
        entry["aliases"] = term.aliases
    if term.meaning:
        entry["meaning"] = term.meaning
    if term.term_type:
        entry["term_type"] = term.term_type
    if term.domain:
        entry["domain"] = term.domain
    return entry


def build_user_prompt(
    text_chunk: str,
    sudachi_candidates: list[SudachiCandidate],
    glossary: Glossary,
    domain: str | None = None,
    max_glossary_terms: int = 50,
) -> str:
    sections: list[str] = []
    if domain:
        sections.append(f"## 対象ドメイン\n{domain}")

    relevant_terms = select_relevant_terms(
        text_chunk=text_chunk,
        sudachi_candidates=sudachi_candidates,
        glossary=glossary,
        max_terms=max_glossary_terms,
    )
    if relevant_terms:
        active_terms = [_format_term_for_prompt(t) for t in relevant_terms]
        sections.append(
            "## 現在の専門用語集\n"
            + json.dumps(active_terms, ensure_ascii=False, indent=2)
        )

    denied = select_relevant_denied_surfaces(
        text_chunk=text_chunk,
        sudachi_candidates=sudachi_candidates,
        glossary=glossary,
    )
    if denied:
        sections.append(f"## 却下済み用語リスト (抽出対象外)\n{', '.join(denied)}")

    cand_lines = [
        f"- {c.surface} (読み: {c.yomi}, 種別: {'OOV' if c.is_oov else '複合名詞'}, 出現回数: {c.count})"
        for c in sudachi_candidates[:50]
    ]
    sections.append(
        f"## Sudachi抽出候補 (ヒント)\n{chr(10).join(cand_lines) if cand_lines else '（なし）'}"
    )
    sections.append(f"## 入力テキスト\n```text\n{text_chunk}\n```")
    return "\n\n".join(sections)


class SystemTwoClient:
    """
    OpenAI 互換の生成型LLM（System Two モデル）を用いて、
    テキストから文脈を考慮した専門用語抽出・定義生成・構造化を行うクライアント。
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        compact_mode: bool = False,
        disable_thinking: bool = True,
        max_glossary_terms: int = 50,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = LLMConfig.from_env(
                model=model,
                api_base=api_base,
                api_key=api_key,
                compact_mode=compact_mode,
                disable_thinking=disable_thinking,
                max_glossary_terms=max_glossary_terms,
            )
        self.client = OpenAI(base_url=self.config.api_base, api_key=self.config.api_key)
        self.call_count = 0

    @classmethod
    def from_config(cls, config: LLMConfig) -> SystemTwoClient:
        return cls(config=config)

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        compact_mode: bool = False,
        disable_thinking: bool = True,
        max_glossary_terms: int = 50,
    ) -> SystemTwoClient:
        return cls(
            config=LLMConfig.from_env(
                model=model,
                api_base=api_base,
                api_key=api_key,
                compact_mode=compact_mode,
                disable_thinking=disable_thinking,
                max_glossary_terms=max_glossary_terms,
            )
        )

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def api_base(self) -> str | None:
        return self.config.api_base

    @property
    def api_key(self) -> str:
        return self.config.api_key

    @property
    def compact_mode(self) -> bool:
        return self.config.compact_mode

    @property
    def disable_thinking(self) -> bool:
        return bool(self.config.disable_thinking)

    @property
    def max_glossary_terms(self) -> int:
        return self.config.max_glossary_terms

    def extract_candidates(
        self,
        text_chunk: str,
        sudachi_candidates: list[SudachiCandidate],
        glossary: Glossary,
        domain: str | None = None,
        sentences: list[str] | None = None,
    ) -> ExtractionResponse:
        """
        テキストチャンクからLLMを用いて専門用語候補を抽出し、
        Sudachi候補メタデータ（読み・用例・ドメイン）で補完したレスポンスを返す。
        """
        user_prompt = build_user_prompt(
            text_chunk=text_chunk,
            sudachi_candidates=sudachi_candidates,
            glossary=glossary,
            domain=domain,
            max_glossary_terms=self.max_glossary_terms,
        )

        sys_prompt = COMPACT_SYSTEM_PROMPT if self.compact_mode else SYSTEM_PROMPT
        resp_format = (
            CompactExtractionResponse if self.compact_mode else TextExtractionResponse
        )

        self.call_count += 1

        if self.disable_thinking:
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=resp_format,
                    temperature=0.1,
                    reasoning_effort="none",
                )
            except (BadRequestError, TypeError):
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=resp_format,
                    temperature=0.1,
                )
        else:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=resp_format,
                temperature=0.1,
            )

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("LLMからの構造化出力のパースに失敗しました。")
        response = parsed.to_extraction_response()
        enrich_proposals_from_sudachi(
            proposals=response.proposals,
            sudachi_candidates=sudachi_candidates,
            sentences=sentences,
            domain=domain,
        )
        return response
