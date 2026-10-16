from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC).astimezone()


def generate_term_id() -> str:
    return f"term-{uuid.uuid4().hex[:8]}"


class TermStatus(StrEnum):
    APPROVED = "approved"
    PENDING = "pending"
    DENIED = "denied"


class PendingUpdate(BaseModel):
    """既存の承認済み用語（status: approved）に対する追加提案（別表記・誤認識など）"""

    aliases: list[str] = Field(
        default_factory=list, description="追加提案された類義語・別表記"
    )
    recognized_texts: list[str] = Field(
        default_factory=list,
        description="追加提案されたOCR・音声文字起こしの誤認識表記",
    )
    examples: list[str] = Field(default_factory=list, description="抽出元の例文・文脈")
    reason: str = Field(default="", description="AIによる追加提案の理由")

    def has_items(self) -> bool:
        return bool(self.aliases or self.recognized_texts)


class Term(BaseModel):
    """専門用語集の1エントリを表すモデル"""

    id: str = Field(default_factory=generate_term_id, description="一意な識別子")
    canonical_text: str = Field(..., description="正式採用する表記")
    yomi: str = Field(default="", description="カタカナ読み（例: キントーン）")
    aliases: list[str] = Field(default_factory=list, description="類義語・別表記の一覧")
    meaning: str = Field(default="", description="用語の意味・解説")
    recognized_texts: list[str] = Field(
        default_factory=list, description="OCRや音声文字起こしで発生した誤認識の一覧"
    )
    term_type: str = Field(default="", description="用語の種類・カテゴリ")
    domain: str = Field(
        default="", description="会社の部署名や特定のプロジェクト名など"
    )
    status: TermStatus = Field(
        default=TermStatus.PENDING,
        description="AIの提案の承認状態 (approved / pending / denied)",
    )
    examples: list[str] = Field(
        default_factory=list, description="抽出元のテキストにおける出現文脈・例文"
    )
    pending_updates: PendingUpdate | None = Field(
        default=None, description="承認済み用語に対する未承認の追記提案"
    )
    created_at: datetime = Field(default_factory=utc_now, description="作成日時")
    updated_at: datetime = Field(default_factory=utc_now, description="最終更新日時")

    def all_surface_forms(self, include_pending: bool = True) -> set[str]:
        forms = {self.canonical_text, *self.aliases, *self.recognized_texts}
        if include_pending and self.pending_updates:
            forms.update(self.pending_updates.aliases)
            forms.update(self.pending_updates.recognized_texts)
        return {f for f in forms if f}

    def has_pending_review(self) -> bool:
        if self.status == TermStatus.PENDING:
            return True
        return bool(
            self.status == TermStatus.APPROVED
            and self.pending_updates is not None
            and self.pending_updates.has_items()
        )

    def apply_pending_updates(self) -> None:
        """pending_updates の内容を本体にマージしてクリアする"""
        if not self.pending_updates:
            return
        for alias in self.pending_updates.aliases:
            if alias and alias != self.canonical_text and alias not in self.aliases:
                self.aliases.append(alias)
        for rec in self.pending_updates.recognized_texts:
            if rec and rec != self.canonical_text and rec not in self.recognized_texts:
                self.recognized_texts.append(rec)
        for ex in self.pending_updates.examples:
            if ex and ex not in self.examples:
                self.examples.append(ex)
        self.pending_updates = None
        self.updated_at = utc_now()

    def discard_pending_updates(self) -> None:
        """pending_updates を破棄する"""
        self.pending_updates = None
        self.updated_at = utc_now()


class SudachiCandidate(BaseModel):
    """Sudachiによる形態素解析で抽出された未知語(OOV)または複合名詞候補"""

    surface: str = Field(..., description="表層形")
    normalized_form: str = Field(default="", description="正規化表記")
    yomi: str = Field(default="", description="カタカナ読み")
    pos: str = Field(default="", description="品詞情報")
    is_oov: bool = Field(default=False, description="辞書未登録語(OOV)かどうか")
    is_compound: bool = Field(default=False, description="複合名詞かどうか")
    count: int = Field(default=1, description="テキスト内の出現回数")
    examples: list[str] = Field(default_factory=list, description="出現した文の抜粋")


class CandidateProposal(BaseModel):
    """LLMが既存の専門用語集と照合して出力する追加・更新提案"""

    proposal_type: Literal["new_term", "update_existing"] = Field(
        default="new_term",
        description="新規用語の追加('new_term') または 既存用語への別表記・誤認識の追加('update_existing')",
    )
    canonical_text: str = Field(
        ...,
        description="正式採用する表記（update_existingの場合は対象となる既存用語のcanonical_text）",
    )
    yomi: str = Field(default="", description="カタカナ読み（例: キントーン）")
    aliases: list[str] = Field(
        default_factory=list, description="類義語・別表記のリスト"
    )
    recognized_texts: list[str] = Field(
        default_factory=list,
        description="OCRや音声文字起こしで発生した誤認識表記のリスト",
    )
    meaning: str = Field(default="", description="文脈から推測される用語の意味・解説")
    term_type: str = Field(
        default="",
        description="用語の種類・カテゴリ（例: product, system, acronym, technical_term, organization, workflow など）",
    )
    domain: str = Field(default="", description="部署名・プロジェクト名・業務領域など")
    examples: list[str] = Field(
        default_factory=list,
        description="テキスト内でその用語や誤認識が使われている例文",
    )
    reason: str = Field(default="", description="専門用語または誤認識と判断した理由")


class TextExtractionProposal(BaseModel):
    """テキストからの専門用語抽出用LLMプロポーザル（クリーンテキスト抽出のため誤認識は含まない）"""

    proposal_type: Literal["new_term", "update_existing"] = Field(
        default="new_term",
        description="新規用語の追加('new_term') または 既存用語への別表記・略称の追加('update_existing')",
    )
    canonical_text: str = Field(
        ...,
        description="正式採用する表記（update_existingの場合は対象となる既存用語のcanonical_text）",
    )
    yomi: str = Field(default="", description="カタカナ読み（例: キントーン）")
    aliases: list[str] = Field(
        default_factory=list, description="類義語・別表記・略称のリスト"
    )
    meaning: str = Field(default="", description="文脈から推測される用語の意味・解説")
    term_type: str = Field(
        default="",
        description="用語の種類・カテゴリ（例: product, system, acronym, technical_term, organization, workflow など）",
    )
    domain: str = Field(default="", description="部署名・プロジェクト名・業務領域など")
    examples: list[str] = Field(
        default_factory=list, description="テキスト内でその用語が使われている例文"
    )
    reason: str = Field(default="", description="専門用語と判断した理由")

    def to_candidate_proposal(self) -> CandidateProposal:
        return CandidateProposal(
            proposal_type=self.proposal_type,
            canonical_text=self.canonical_text,
            yomi=self.yomi,
            aliases=self.aliases,
            recognized_texts=[],
            meaning=self.meaning,
            term_type=self.term_type,
            domain=self.domain,
            examples=self.examples,
            reason=self.reason,
        )


class TextExtractionResponse(BaseModel):
    """LLMへの構造化出力用レスポンス（クリーンテキストからの専門用語抽出用、誤認識は除外）"""

    proposals: list[TextExtractionProposal] = Field(
        default_factory=list, description="抽出された専門用語・更新提案のリスト"
    )

    def to_extraction_response(self) -> ExtractionResponse:
        return ExtractionResponse(
            proposals=[p.to_candidate_proposal() for p in self.proposals]
        )


class ExtractionResponse(BaseModel):
    """LLMからの構造化出力レスポンス"""

    proposals: list[CandidateProposal] = Field(
        default_factory=list, description="抽出された専門用語・更新提案のリスト"
    )


class Glossary(BaseModel):
    """専門用語集全体を管理するコンテナモデル"""

    terms: list[Term] = Field(
        default_factory=list, description="登録されている専門用語リスト"
    )

    def find_by_canonical(self, canonical_text: str) -> Term | None:
        normalized = canonical_text.strip()
        for term in self.terms:
            if term.canonical_text == normalized:
                return term
        return None

    def find_by_surface(self, surface: str) -> Term | None:
        normalized = surface.strip()
        for term in self.terms:
            if normalized in term.all_surface_forms(include_pending=True):
                return term
        return None

    def get_pending_reviews(self) -> list[Term]:
        return [t for t in self.terms if t.has_pending_review()]

    def merge_proposal(
        self, proposal: CandidateProposal, default_domain: str | None = None
    ) -> tuple[Term | None, str]:
        """
        AIからの提案をGlossaryにマージする。
        戻り値: (対象TermまたはNone, アクション種別: 'added' | 'updated_pending' | 'merged_pending' | 'skipped_denied' | 'unchanged')
        """
        canonical = proposal.canonical_text.strip()
        if not canonical:
            return None, "unchanged"

        # 既存用語の検索（canonical_text または surface 一致）
        existing = self.find_by_canonical(canonical) or self.find_by_surface(canonical)

        # フォールバック: LLMが誤って誤認識語を canonical_text にし、本来の用語を aliases に入れた場合
        if existing is None:
            for alias in proposal.aliases:
                alias_clean = alias.strip()
                if not alias_clean:
                    continue
                found = self.find_by_canonical(alias_clean) or self.find_by_surface(
                    alias_clean
                )
                if found is not None:
                    existing = found
                    is_misrec = (
                        "誤認識" in proposal.meaning
                        or "誤認識" in proposal.reason
                        or "誤変換" in proposal.meaning
                        or "誤変換" in proposal.reason
                    )
                    if is_misrec:
                        if canonical not in proposal.recognized_texts:
                            proposal.recognized_texts.append(canonical)
                    elif canonical not in proposal.aliases:
                        proposal.aliases.append(canonical)
                    break

        if existing is not None:
            if existing.status == TermStatus.DENIED:
                return existing, "skipped_denied"

            new_aliases = [
                a.strip()
                for a in proposal.aliases
                if a.strip()
                and a.strip() != existing.canonical_text
                and a.strip() not in existing.aliases
            ]
            new_rec = [
                r.strip()
                for r in proposal.recognized_texts
                if r.strip()
                and r.strip() != existing.canonical_text
                and r.strip() not in existing.recognized_texts
            ]
            # もし proposal.canonical_text が既存用語の canonical_text と異なり、まだ登録されていなければ aliases 扱いにする
            if (
                canonical != existing.canonical_text
                and canonical not in existing.aliases
                and canonical not in existing.recognized_texts
                and canonical not in new_aliases
                and canonical not in new_rec
            ):
                new_aliases.append(canonical)

            new_examples = [
                ex.strip()
                for ex in proposal.examples
                if ex.strip() and ex.strip() not in existing.examples
            ]

            if existing.status == TermStatus.PENDING:
                # まだ pending の用語なら直接マージする
                changed = False
                for a in new_aliases:
                    if a not in existing.aliases:
                        existing.aliases.append(a)
                        changed = True
                for r in new_rec:
                    if r not in existing.recognized_texts:
                        existing.recognized_texts.append(r)
                        changed = True
                for ex in new_examples:
                    if ex not in existing.examples:
                        existing.examples.append(ex)
                        changed = True
                if not existing.yomi and proposal.yomi:
                    existing.yomi = proposal.yomi
                    changed = True
                if not existing.meaning and proposal.meaning:
                    existing.meaning = proposal.meaning
                    changed = True
                if not existing.term_type and proposal.term_type:
                    existing.term_type = proposal.term_type
                    changed = True
                if not existing.domain and (proposal.domain or default_domain):
                    existing.domain = proposal.domain or default_domain or ""
                    changed = True
                if changed:
                    existing.updated_at = utc_now()
                    return existing, "merged_pending"
                return existing, "unchanged"

            # existing.status == TermStatus.APPROVED の場合は pending_updates に格納する
            if existing.pending_updates is None:
                existing.pending_updates = PendingUpdate()

            added_to_pending = False
            for a in new_aliases:
                if a not in existing.pending_updates.aliases:
                    existing.pending_updates.aliases.append(a)
                    added_to_pending = True
            for r in new_rec:
                if r not in existing.pending_updates.recognized_texts:
                    existing.pending_updates.recognized_texts.append(r)
                    added_to_pending = True
            for ex in new_examples:
                if ex not in existing.pending_updates.examples:
                    existing.pending_updates.examples.append(ex)
                    added_to_pending = True

            if added_to_pending:
                if proposal.reason and not existing.pending_updates.reason:
                    existing.pending_updates.reason = proposal.reason
                existing.updated_at = utc_now()
                return existing, "updated_pending"
            else:
                if not existing.pending_updates.has_items():
                    existing.pending_updates = None
                return existing, "unchanged"

        # 新規用語として追加
        clean_aliases = [
            a.strip() for a in proposal.aliases if a.strip() and a.strip() != canonical
        ]
        clean_rec = [
            r.strip()
            for r in proposal.recognized_texts
            if r.strip() and r.strip() != canonical
        ]
        clean_examples = [ex.strip() for ex in proposal.examples if ex.strip()]

        new_term = Term(
            canonical_text=canonical,
            yomi=proposal.yomi.strip(),
            aliases=list(dict.fromkeys(clean_aliases)),
            recognized_texts=list(dict.fromkeys(clean_rec)),
            meaning=proposal.meaning.strip(),
            term_type=proposal.term_type.strip(),
            domain=(proposal.domain.strip() or default_domain or ""),
            status=TermStatus.PENDING,
            examples=list(dict.fromkeys(clean_examples)),
        )
        self.terms.append(new_term)
        return new_term, "added"
