from __future__ import annotations

import json
import urllib.request
from typing import Any

from jjargon.models import (
    CandidateProposal,
    ExtractionResponse,
    Glossary,
    SudachiCandidate,
    SystemOneConfig,
    Term,
    TermStatus,
    normalize_key,
)

TERM_TYPE_CRITERIA: dict[str, str] = {
    "technical_term": "技術用語・専門概念・手法",
    "business_term": "社内業務・経営・制度の専門用語",
    "system": "社内システム・IT基盤",
    "product": "製品名・商用サービス・SaaS",
    "acronym": "アルファベット略語・頭字語",
    "organization": "組織名・部署名・企業名・委員会",
    "workflow": "業務フロー・申請手続き",
}


def _is_candidate_related_to_term(cand: SudachiCandidate, term: Term) -> bool:
    """候補が既存用語の別表記・略称である可能性があるか（読み一致または2文字以上の共通部分文字列）を高速判定"""
    norm_surf = normalize_key(cand.surface)
    norm_yomi = normalize_key(cand.yomi) if cand.yomi else ""
    term_yomi = normalize_key(term.yomi) if term.yomi else ""

    if (
        len(norm_yomi) >= 3
        and len(term_yomi) >= 3
        and (norm_yomi == term_yomi or norm_yomi in term_yomi or term_yomi in norm_yomi)
    ):
        return True

    for form in term.all_surface_forms(include_pending=True):
        nf = normalize_key(form)
        if len(norm_surf) >= 2 and len(nf) >= 2:
            shorter, longer = (
                (norm_surf, nf) if len(norm_surf) <= len(nf) else (nf, norm_surf)
            )
            if any(shorter[i : i + 2] in longer for i in range(len(shorter) - 1)):
                return True
    return False


def prefilter_sudachi_candidates(
    candidates: list[SudachiCandidate],
    glossary: Glossary | None = None,
    max_candidates: int = 16,
) -> list[SudachiCandidate]:
    """
    System One に投げる前に、機械的に除外可能なノイズ候補を高速フィルタリングする：
    1. 既にGlossaryに登録済みの表層形
    2. 1文字のみの候補（漢数字「一」など）
    3. Sudachiの長単位辞書(SplitMode.C)に既に1語として登録されている一般複合語 (`pos == '複合名詞(C)'` かつ非OOV)
    4. 同一チャンク内のより長い候補の部分文字列になっている短い重複候補（例: 「テラパワー社」に対する「テラパワー」）
    """
    known_surfaces: set[str] = set()
    if glossary is not None:
        for term in glossary.terms:
            for form in term.all_surface_forms(include_pending=True):
                nf = normalize_key(form)
                if nf:
                    known_surfaces.add(nf)

    oov_surfaces = {c.surface for c in candidates if c.is_oov and len(c.surface) >= 2}
    suffix_tokens = ("上", "内", "中", "後", "前", "下", "外", "間", "毎", "別", "等")

    step1: list[SudachiCandidate] = []
    for c in candidates:
        s = c.surface.strip()
        if len(s) < 2:
            continue
        if normalize_key(s) in known_surfaces:
            continue
        # SplitMode.C で既に辞書に1単語として載っている既知語（複合名詞(C)）は除外（ただし全候補が複合名詞(C)のみの場合は残す）
        if not c.is_oov and c.pos == "複合名詞(C)":
            continue
        if c.is_compound and s.endswith(suffix_tokens) and s[:-1] in oov_surfaces:
            continue
        if c.is_compound and s.endswith("等"):
            continue
        step1.append(c)

    # もし step1 が空（テスト用の短い文で複合名詞(C)しか無い場合など）は複合名詞(C)も許容するフォールバック
    if not step1:
        for c in candidates:
            s = c.surface.strip()
            if len(s) >= 2 and normalize_key(s) not in known_surfaces:
                step1.append(c)

    # 同一チャンク内で「テラパワー社」と「テラパワー」のように一方が他方の部分文字列で出現回数が同じ場合は長い方を優先
    deduped: list[SudachiCandidate] = []
    for c in step1:
        is_redundant_sub = any(
            other.surface != c.surface
            and c.surface in other.surface
            and other.count >= c.count
            for other in step1
        )
        if not is_redundant_sub:
            deduped.append(c)

    return deduped[:max_candidates]


class SystemOneClient:
    """
    Ollama /v1/systemone (Kev) や TypeSafe Jev などの System One モデルを用いて、
    テキスト生成を行わずに高速な専門用語判定(noul)・種別分類(choice)・既存用語名寄せ(choice)を行うクライアント。
    """

    def __init__(
        self,
        config: SystemOneConfig | None = None,
        *,
        model: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        threshold: float = 0.75,
        batch_size: int = 16,
        timeout: float = 60.0,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = SystemOneConfig.from_env(
                model=model,
                url=url,
                api_key=api_key,
                threshold=threshold,
                batch_size=batch_size,
                timeout=timeout,
            )
        self.call_count = 0

    @classmethod
    def from_config(cls, config: SystemOneConfig) -> SystemOneClient:
        return cls(config=config)

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        threshold: float = 0.75,
        batch_size: int = 16,
        timeout: float = 60.0,
    ) -> SystemOneClient:
        return cls(
            config=SystemOneConfig.from_env(
                model=model,
                url=url,
                api_key=api_key,
                threshold=threshold,
                batch_size=batch_size,
                timeout=timeout,
            )
        )

    @property
    def model(self) -> str:
        return self.config.model or ""

    @property
    def url(self) -> str:
        return self.config.url or ""

    @property
    def api_key(self) -> str | None:
        return self.config.api_key

    @property
    def threshold(self) -> float:
        return self.config.threshold

    @property
    def batch_size(self) -> int:
        return self.config.batch_size

    @property
    def timeout(self) -> float:
        return self.config.timeout

    def call(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        """/v1/systemone エンドポイントを呼び出して answers 辞書を返す"""
        if not questions:
            return {}
        payload = {
            "model": self.model,
            "state": state,
            "questions": questions,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key.strip():
            key = self.api_key.strip()
            headers["Authorization"] = (
                key if key.startswith("Bearer ") else f"Bearer {key}"
            )
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        self.call_count += 1
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        answers = body.get("answers", {})
        return answers if isinstance(answers, dict) else {}

    def filter_candidates(
        self,
        candidates: list[SudachiCandidate],
        glossary: Glossary | None = None,
        domain: str | None = None,
    ) -> tuple[list[SudachiCandidate], dict[str, float]]:
        """
        Sudachi候補リストに対して事前機械フィルタ＋System One の noul 判定を実行し、
        専門用語である確率が self.threshold 以上の候補のみに絞り込む。
        """
        if not candidates:
            return [], {}

        target_candidates = prefilter_sudachi_candidates(
            candidates=candidates,
            glossary=glossary,
            max_candidates=self.batch_size,
        )
        if not target_candidates:
            return [], {}

        scores: dict[str, float] = {}
        filtered: list[SudachiCandidate] = []

        for start_idx in range(0, len(target_candidates), self.batch_size):
            batch = target_candidates[start_idx : start_idx + self.batch_size]
            cand_state: dict[str, dict[str, str]] = {}
            questions: dict[str, Any] = {}

            for i, cand in enumerate(batch):
                cid = f"c{i}"
                cand_state[cid] = {"surface": cand.surface}
                ex_str = f"（文脈: {cand.examples[0]}）" if cand.examples else ""
                questions[f"{cid}_is_jargon"] = {
                    "type": "noul",
                    "instructions": (
                        f"語句「{cand.surface}」{ex_str}は、一般的な日常語・一般ビジネス語ではなく、"
                        "専門用語集に登録すべき固有の専門用語・社内独自用語・IT製品名・システム名・略称か？"
                    ),
                    "criteria": {
                        "true": "固有の専門用語・技術用語・社内独自用語・製品名・システム名・プロジェクト名・略称",
                        "false": "一般的な日常語・一般ビジネス語（確認作業、資料作成、部門予算、年末調整、運転開始、技術協力、共同研究開発、世紀半ばなど）",
                    },
                }

            state: dict[str, Any] = {"candidates": cand_state}
            if domain:
                state["domain"] = domain

            answers = self.call(state=state, questions=questions)
            for i, cand in enumerate(batch):
                cid = f"c{i}"
                ans = answers.get(f"{cid}_is_jargon", {})
                noul_score = (
                    float(ans.get("noul", 0.0)) if isinstance(ans, dict) else 0.0
                )
                scores[cand.surface] = noul_score
                if noul_score >= self.threshold:
                    filtered.append(cand)

        return filtered, scores

    def classify_candidates(
        self,
        candidates: list[SudachiCandidate],
        glossary: Glossary,
        relevant_terms: list[Term] | None = None,
        domain: str | None = None,
        scores: dict[str, float] | None = None,
    ) -> ExtractionResponse:
        """
        System One 完結モード用:
        フィルタ済みのSudachi候補に対して、type: 'choice' で
        1. 用語種別 (term_type)
        2. 既存用語との名寄せ (update_existing vs new_term)
        を一括判定し、Sudachiの yomi と examples を付与して ExtractionResponse を返す。
        """
        if not candidates:
            return ExtractionResponse(proposals=[])

        active_terms = [
            t
            for t in (relevant_terms if relevant_terms is not None else glossary.terms)
            if t.status != TermStatus.DENIED
        ]

        term_map: dict[str, Term] = {}
        existing_criteria: dict[str, str] = {}
        existing_state_list: list[dict[str, Any]] = []
        for idx, t in enumerate(active_terms[:15]):
            t_key = f"term_{idx}"
            term_map[t_key] = t
            aliases_str = f", 別表記: {', '.join(t.aliases)}" if t.aliases else ""
            existing_criteria[t_key] = (
                f"既存用語「{t.canonical_text}」(読み: {t.yomi}{aliases_str}) の別表記・略称である"
            )
            existing_state_list.append(
                {
                    "key": t_key,
                    "canonical_text": t.canonical_text,
                    "yomi": t.yomi,
                }
            )
        existing_criteria["new_term"] = (
            "既存用語のいずれにも該当しない新規の専門用語である"
        )

        proposals: list[CandidateProposal] = []

        for start_idx in range(0, len(candidates), self.batch_size):
            batch = candidates[start_idx : start_idx + self.batch_size]
            cand_state: dict[str, dict[str, str]] = {}
            questions: dict[str, Any] = {}
            has_existing_question = False

            for i, cand in enumerate(batch):
                cid = f"c{i}"
                cand_state[cid] = {"surface": cand.surface}
                ex_str = f"（文脈: {cand.examples[0]}）" if cand.examples else ""
                questions[f"{cid}_term_type"] = {
                    "type": "choice",
                    "instructions": f"語句「{cand.surface}」{ex_str}の用語種別を選択せよ。",
                    "criteria": TERM_TYPE_CRITERIA,
                }
                if term_map and any(
                    _is_candidate_related_to_term(cand, t) for t in term_map.values()
                ):
                    has_existing_question = True
                    questions[f"{cid}_existing_match"] = {
                        "type": "choice",
                        "instructions": (
                            f"語句「{cand.surface}」{ex_str}は existing_glossary 内の既存用語の別表記・略称か、"
                            "それとも新規の専門用語か？"
                        ),
                        "criteria": existing_criteria,
                    }

            state: dict[str, Any] = {"candidates": cand_state}
            if domain:
                state["domain"] = domain
            if has_existing_question and existing_state_list:
                state["existing_glossary"] = existing_state_list

            answers = self.call(state=state, questions=questions)

            for i, cand in enumerate(batch):
                cid = f"c{i}"
                type_ans = answers.get(f"{cid}_term_type", {})
                term_type = (
                    str(type_ans.get("choice", "technical_term"))
                    if isinstance(type_ans, dict)
                    else "technical_term"
                )

                match_choice = "new_term"
                if f"{cid}_existing_match" in questions:
                    match_ans = answers.get(f"{cid}_existing_match", {})
                    if isinstance(match_ans, dict):
                        match_choice = str(match_ans.get("choice", "new_term"))

                score_val = (scores or {}).get(cand.surface)
                reason_str = (
                    f"System One ({self.model}) 抽出 (score={score_val:.2f})"
                    if score_val is not None
                    else f"System One ({self.model}) 抽出"
                )

                if match_choice in term_map:
                    matched_term = term_map[match_choice]
                    proposals.append(
                        CandidateProposal(
                            proposal_type="update_existing",
                            canonical_text=matched_term.canonical_text,
                            yomi=matched_term.yomi or cand.yomi,
                            aliases=[cand.surface],
                            meaning="",
                            term_type=matched_term.term_type or term_type,
                            domain=domain or matched_term.domain or "",
                            examples=list(cand.examples),
                            reason=reason_str,
                        )
                    )
                else:
                    proposals.append(
                        CandidateProposal(
                            proposal_type="new_term",
                            canonical_text=cand.surface,
                            yomi=cand.yomi,
                            aliases=[],
                            meaning="",
                            term_type=term_type,
                            domain=domain or "",
                            examples=list(cand.examples),
                            reason=reason_str,
                        )
                    )

        return ExtractionResponse(proposals=proposals)
