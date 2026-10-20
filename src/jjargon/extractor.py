from __future__ import annotations

import os
from collections.abc import Callable
from enum import StrEnum

from jjargon.analyzer import SudachiAnalyzer
from jjargon.models import (
    CandidateProposal,
    ExtractionResponse,
    Glossary,
    SudachiCandidate,
    Term,
    enrich_proposals_from_sudachi,
)
from jjargon.systemone import SystemOneClient, prefilter_sudachi_candidates
from jjargon.systemtwo import (
    COMPACT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    SystemTwoClient,
    build_user_prompt,
    select_relevant_denied_surfaces,
    select_relevant_terms,
)

__all__ = [
    "COMPACT_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "ExtractionMode",
    "JargonExtractor",
    "SystemOneClient",
    "SystemTwoClient",
    "build_user_prompt",
    "select_relevant_denied_surfaces",
    "select_relevant_terms",
]


class ExtractionMode(StrEnum):
    """専門用語抽出パイプラインの実行モード"""

    HYBRID = "hybrid"  # System 1 でトリアージ・候補絞り込み -> System 2 で生成抽出
    SYSTEM_ONE_ONLY = "system_one_only"  # System 1 のみで判定・分類抽出 (LLM生成不使用・最速)
    SYSTEM_TWO_ONLY = "system_two_only"  # System 2 のみで生成抽出 (System 1 不使用)


class JargonExtractor:
    """
    Sudachi形態素解析、System One (判定・分類モデル)、および System Two (生成LLM) を
    対等なコンポーネントとして統合し、指定された ExtractionMode に従って専門用語を抽出するオーケストレーター。
    """

    def __init__(
        self,
        mode: ExtractionMode | str = ExtractionMode.HYBRID,
        system_two: SystemTwoClient | None = None,
        system_one: SystemOneClient | None = None,
        max_chunk_chars: int = 4000,
        analyzer: SudachiAnalyzer | None = None,
        # 互換性および簡易フラット指定用引数
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        compact_mode: bool = False,
        disable_thinking: bool | None = None,
        max_glossary_terms: int = 50,
        systemone_model: str | None = None,
        systemone_url: str | None = None,
        systemone_threshold: float = 0.75,
        # 従来のブールフラグエイリアス
        systemone_only: bool = False,
        use_systemone: bool | None = None,
        systemone_client: SystemOneClient | None = None,
    ) -> None:
        self.analyzer = analyzer or SudachiAnalyzer()
        self.max_chunk_chars = max_chunk_chars

        # 1. 実行モードの解決
        if systemone_only:
            self.mode = ExtractionMode.SYSTEM_ONE_ONLY
        elif use_systemone is False:
            self.mode = ExtractionMode.SYSTEM_TWO_ONLY
        elif isinstance(mode, str):
            self.mode = ExtractionMode(mode)
        else:
            self.mode = mode

        # 2. System Two (生成LLM) クライアントの解決
        if self.mode in (ExtractionMode.HYBRID, ExtractionMode.SYSTEM_TWO_ONLY):
            if system_two is not None:
                self.system_two: SystemTwoClient | None = system_two
            else:
                self.system_two = SystemTwoClient.from_env(
                    model=model,
                    api_base=api_base,
                    api_key=api_key,
                    compact_mode=compact_mode,
                    disable_thinking=compact_mode if disable_thinking is None else disable_thinking,
                    max_glossary_terms=max_glossary_terms,
                )
        else:
            self.system_two = None

        # 3. System One (意思決定モデル) クライアントの解決
        if self.mode in (ExtractionMode.HYBRID, ExtractionMode.SYSTEM_ONE_ONLY):
            s1 = system_one or systemone_client
            if s1 is not None:
                self.system_one: SystemOneClient | None = s1
            else:
                env_s1_model = os.environ.get("SYSTEMONE_MODEL")
                env_s1_url = os.environ.get("SYSTEMONE_URL")
                has_s1_input = bool(systemone_model or systemone_url or env_s1_model or env_s1_url)

                if self.mode == ExtractionMode.SYSTEM_ONE_ONLY or use_systemone is True or has_s1_input:
                    # 明示的な指定または環境変数がある場合は初期化（不足時はSystemOneClientがValueErrorを送出）
                    self.system_one = SystemOneClient.from_env(
                        model=systemone_model,
                        url=systemone_url,
                        threshold=systemone_threshold,
                    )
                else:
                    # HYBRID 指定だが System 1 の設定が一切無い場合は、安全に System Two Only として動作
                    self.system_one = None
                    self.mode = ExtractionMode.SYSTEM_TWO_ONLY
        else:
            self.system_one = None

        self.last_stats: dict[str, int] = {
            "total_chunks": 0,
            "llm_calls": 0,
            "systemone_calls": 0,
            "skipped_chunks": 0,
            "candidates_total": 0,
            "candidates_filtered": 0,
        }

    # 後方互換性プロパティ
    @property
    def model(self) -> str:
        return self.system_two.model if self.system_two else ""

    @property
    def api_base(self) -> str | None:
        return self.system_two.api_base if self.system_two else None

    @property
    def api_key(self) -> str:
        return self.system_two.api_key if self.system_two else ""

    @property
    def client(self):
        return self.system_two.client if self.system_two else None

    @property
    def compact_mode(self) -> bool:
        return self.system_two.compact_mode if self.system_two else False

    @property
    def max_glossary_terms(self) -> int:
        return self.system_two.max_glossary_terms if self.system_two else 50

    @property
    def systemone_client(self) -> SystemOneClient | None:
        return self.system_one

    @property
    def systemone_only(self) -> bool:
        return self.mode == ExtractionMode.SYSTEM_ONE_ONLY

    def extract_candidates(
        self,
        text_chunk: str,
        sudachi_candidates: list[SudachiCandidate],
        glossary: Glossary,
        domain: str | None = None,
    ) -> ExtractionResponse:
        """System Two (LLM) を用いてチャンクから専門用語候補を抽出する"""
        if self.system_two is None:
            raise RuntimeError(
                f"現在のモード ({self.mode}) では System Two クライアントが無効です。"
            )
        return self.system_two.extract_candidates(
            text_chunk=text_chunk,
            sudachi_candidates=sudachi_candidates,
            glossary=glossary,
            domain=domain,
            sentences=self.analyzer.split_sentences(text_chunk),
        )

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
        on_progress: Callable[[int, int], None] | None = None,
    ) -> tuple[list[CandidateProposal], dict[str, list[Term]]]:
        """テキスト全体をチャンク分割し、現在のモードに従って用語集へマージする"""
        chunks = self.chunk_text(text)
        total_chunks = len(chunks)
        self.last_stats = {
            "total_chunks": total_chunks,
            "llm_calls": 0,
            "systemone_calls": 0,
            "skipped_chunks": 0,
            "candidates_total": 0,
            "candidates_filtered": 0,
        }
        if self.system_one is not None:
            self.system_one.call_count = 0
        if self.system_two is not None:
            self.system_two.call_count = 0

        if on_progress is not None:
            on_progress(0, total_chunks)

        all_proposals: list[CandidateProposal] = []
        summary: dict[str, list[Term]] = {
            "added": [],
            "updated_pending": [],
            "merged_pending": [],
            "skipped_denied": [],
            "unchanged": [],
        }

        for idx, chunk in enumerate(chunks, start=1):
            sudachi_candidates = self.analyzer.analyze(
                text=chunk,
                glossary=glossary,
                include_compounds=include_compounds,
            )
            self.last_stats["candidates_total"] += len(sudachi_candidates)

            chunk_proposals: list[CandidateProposal] = []

            # -------------------------------------------------------------
            # モード 1: SYSTEM_ONE_ONLY (生成LLM不使用・最速)
            # -------------------------------------------------------------
            if self.mode == ExtractionMode.SYSTEM_ONE_ONLY:
                assert self.system_one is not None
                active_cands, scores = self.system_one.filter_candidates(
                    text_chunk=chunk,
                    candidates=sudachi_candidates,
                    glossary=glossary,
                    domain=domain,
                )
                self.last_stats["candidates_filtered"] += len(active_cands)
                self.last_stats["systemone_calls"] = self.system_one.call_count

                if not active_cands:
                    self.last_stats["skipped_chunks"] += 1
                else:
                    relevant_terms = select_relevant_terms(
                        text_chunk=chunk,
                        sudachi_candidates=active_cands,
                        glossary=glossary,
                        max_terms=self.max_glossary_terms,
                    )
                    response = self.system_one.classify_candidates(
                        text_chunk=chunk,
                        candidates=active_cands,
                        glossary=glossary,
                        relevant_terms=relevant_terms,
                        domain=domain,
                        scores=scores,
                    )
                    self.last_stats["systemone_calls"] = self.system_one.call_count
                    enrich_proposals_from_sudachi(
                        proposals=response.proposals,
                        sudachi_candidates=sudachi_candidates,
                        sentences=self.analyzer.split_sentences(chunk),
                        domain=domain,
                    )
                    chunk_proposals = response.proposals

            # -------------------------------------------------------------
            # モード 2: SYSTEM_TWO_ONLY (LLM生成単体)
            # -------------------------------------------------------------
            elif self.mode == ExtractionMode.SYSTEM_TWO_ONLY:
                assert self.system_two is not None
                active_cands = sudachi_candidates
                if self.system_two.compact_mode:
                    active_cands = prefilter_sudachi_candidates(
                        candidates=sudachi_candidates,
                        glossary=glossary,
                        max_candidates=25,
                    )
                response = self.extract_candidates(
                    text_chunk=chunk,
                    sudachi_candidates=active_cands,
                    glossary=glossary,
                    domain=domain,
                )
                self.last_stats["llm_calls"] += 1
                enrich_proposals_from_sudachi(
                    proposals=response.proposals,
                    sudachi_candidates=sudachi_candidates,
                    sentences=self.analyzer.split_sentences(chunk),
                    domain=domain,
                )
                chunk_proposals = response.proposals

            # -------------------------------------------------------------
            # モード 3: HYBRID (System 1 でフィルタ/スキップ -> System 2 で生成)
            # -------------------------------------------------------------
            elif self.mode == ExtractionMode.HYBRID:
                assert self.system_one is not None
                assert self.system_two is not None
                active_cands, _ = self.system_one.filter_candidates(
                    text_chunk=chunk,
                    candidates=sudachi_candidates,
                    glossary=glossary,
                    domain=domain,
                )
                self.last_stats["candidates_filtered"] += len(active_cands)
                self.last_stats["systemone_calls"] = self.system_one.call_count

                if not active_cands:
                    self.last_stats["skipped_chunks"] += 1
                else:
                    response = self.extract_candidates(
                        text_chunk=chunk,
                        sudachi_candidates=active_cands,
                        glossary=glossary,
                        domain=domain,
                    )
                    self.last_stats["llm_calls"] += 1
                    enrich_proposals_from_sudachi(
                        proposals=response.proposals,
                        sudachi_candidates=sudachi_candidates,
                        sentences=self.analyzer.split_sentences(chunk),
                        domain=domain,
                    )
                    chunk_proposals = response.proposals

            # 提案を用語集へマージ
            for proposal in chunk_proposals:
                all_proposals.append(proposal)
                term, action = glossary.merge_proposal(proposal, default_domain=domain)
                if (
                    term is not None
                    and action in summary
                    and term not in summary[action]
                ):
                    summary[action].append(term)

            if on_progress is not None:
                on_progress(idx, total_chunks)

        return all_proposals, summary
