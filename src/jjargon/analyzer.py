import re

from sudachipy import Dictionary, Morpheme, SplitMode

from jjargon.models import Glossary, SudachiCandidate, TermStatus

# 文分割用の正規表現（句点、改行、感嘆符・疑問符など）
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？\n])\s*|\n+")
# URLや単なる数値・記号のみのノイズ判定用正規表現
URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
PURE_NUM_SYMBOL_RE = re.compile(
    r"^[\d\s.,\-+/:;()\[\]{}<>「」『』【】・、。！？!?'\"#$%&*=~^|\\_]+$"
)
SINGLE_HIRAGANA_RE = re.compile(r"^[ぁ-んー]$")
# 「2026年」「10月」「3件」「100円」などの数値＋助数詞パターン
NUMERIC_COUNTER_RE = re.compile(
    r"^[\d０-９一二三四五六七八九十百千万億兆]+(年|月|日|時|分|秒|件|回|人|個|社|名|円|割|%|％|つ|度|番|号|階|歳|週間|ヶ月|か月|カ月)$"
)
MAX_EXAMPLES_PER_CANDIDATE = 3


class SudachiAnalyzer:
    """Sudachiを用いてテキストからOOV（辞書未登録語）および複合名詞候補を抽出するアナライザ"""

    def __init__(self) -> None:
        self._dictionary = Dictionary(dict="core")
        self._tokenizer = self._dictionary.tokenizer()

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        raw_parts = SENTENCE_SPLIT_RE.split(text)
        return [part.strip() for part in raw_parts if part and part.strip()]

    @staticmethod
    def _is_noise_surface(surface: str) -> bool:
        s = surface.strip()
        if not s:
            return True
        if URL_RE.match(s):
            return True
        if PURE_NUM_SYMBOL_RE.match(s):
            return True
        if SINGLE_HIRAGANA_RE.match(s):
            return True
        return bool(NUMERIC_COUNTER_RE.match(s))

    @staticmethod
    def _is_valid_oov_morpheme(m: Morpheme) -> bool:
        if not m.is_oov():
            return False
        pos = m.part_of_speech()
        major_pos = pos[0]
        if major_pos in {"空白", "補助記号", "記号", "助詞", "助動詞", "感動詞"}:
            return False
        surface = m.surface().strip()
        return not SudachiAnalyzer._is_noise_surface(surface)

    @staticmethod
    def _can_be_in_compound(m: Morpheme, position: str) -> bool:
        """
        複合名詞の構成要素になり得る形態素かを判定する。
        position: 'start' | 'middle' | 'end'
        """
        pos = m.part_of_speech()
        major = pos[0] if len(pos) > 0 else ""
        sub1 = pos[1] if len(pos) > 1 else ""
        sub2 = pos[2] if len(pos) > 2 else ""

        surface = m.surface().strip()
        if not surface or PURE_NUM_SYMBOL_RE.match(surface):
            return False

        # 代名詞や非自立語単体、副詞可能のみ等は除外
        if major == "名詞":
            if sub1 in {"代名詞", "助動詞語幹"}:
                return False
            if sub1 == "数詞":
                return False
            return not (
                sub1 == "普通名詞"
                and sub2 in {"副詞可能", "助数詞可能"}
                and position == "start"
            )

        if major == "接頭辞" and position == "start":
            return True

        if major == "接尾辞" and position in {"middle", "end"}:
            return not (sub1 == "名詞的" and sub2 == "助数詞")

        # OOV（英単語や未知のカタカナ語など）も複合名詞の一部として許容
        return bool(m.is_oov() and not SudachiAnalyzer._is_noise_surface(surface))

    def analyze(
        self,
        text: str,
        glossary: Glossary | None = None,
        include_compounds: bool = True,
    ) -> list[SudachiCandidate]:
        """
        テキストを解析し、OOV（未知語）および複合名詞の候補リストを返す。
        glossary が渡された場合、既に status='denied' の語は除外する。
        """
        sentences = self.split_sentences(text)
        candidates_map: dict[str, SudachiCandidate] = {}

        denied_surfaces: set[str] = set()
        if glossary is not None:
            for term in glossary.terms:
                if term.status == TermStatus.DENIED:
                    denied_surfaces.update(
                        term.all_surface_forms(include_pending=False)
                    )

        def register_candidate(
            surface: str,
            normalized: str,
            yomi: str,
            pos_str: str,
            is_oov: bool,
            is_compound: bool,
            sentence: str,
        ) -> None:
            s = surface.strip()
            if self._is_noise_surface(s) or s in denied_surfaces:
                return
            if s in candidates_map:
                cand = candidates_map[s]
                cand.count += 1
                cand.is_oov = cand.is_oov or is_oov
                cand.is_compound = cand.is_compound or is_compound
                if (
                    sentence not in cand.examples
                    and len(cand.examples) < MAX_EXAMPLES_PER_CANDIDATE
                ):
                    cand.examples.append(sentence)
            else:
                candidates_map[s] = SudachiCandidate(
                    surface=s,
                    normalized_form=normalized or s,
                    yomi=yomi,
                    pos=pos_str,
                    is_oov=is_oov,
                    is_compound=is_compound,
                    count=1,
                    examples=[sentence] if sentence else [],
                )

        for sentence in sentences:
            # 1. SplitMode.C（長単位）および SplitMode.A（短単位）で解析
            morphemes_c = self._tokenizer.tokenize(sentence, SplitMode.C)
            morphemes_a = self._tokenizer.tokenize(sentence, SplitMode.A)

            # OOVの抽出（SplitMode.C と SplitMode.A の両方から拾う）
            seen_oov_in_sentence: set[str] = set()
            for m in list(morphemes_c) + list(morphemes_a):
                if self._is_valid_oov_morpheme(m):
                    surf = m.surface().strip()
                    if surf not in seen_oov_in_sentence:
                        seen_oov_in_sentence.add(surf)
                        pos_str = "-".join(
                            p for p in m.part_of_speech()[:2] if p != "*"
                        )
                        register_candidate(
                            surface=surf,
                            normalized=m.normalized_form(),
                            yomi=m.reading_form(),
                            pos_str=pos_str,
                            is_oov=True,
                            is_compound=False,
                            sentence=sentence,
                        )

            if not include_compounds:
                continue

            # 2. 複合名詞の抽出
            # (a) SplitMode.C で1形態素だが SplitMode.A では複数形態素に分かれる名詞
            for mc in morphemes_c:
                pos = mc.part_of_speech()
                if pos[0] == "名詞" and pos[1] != "数詞":
                    sub_a = mc.split(SplitMode.A)
                    if len(sub_a) >= 2:
                        surf = mc.surface().strip()
                        if not self._is_noise_surface(surf):
                            register_candidate(
                                surface=surf,
                                normalized=mc.normalized_form(),
                                yomi=mc.reading_form(),
                                pos_str="複合名詞(C)",
                                is_oov=mc.is_oov(),
                                is_compound=True,
                                sentence=sentence,
                            )

            # (b) SplitMode.A で連続する名詞列の連結（2形態素以上）
            n = len(morphemes_a)
            i = 0
            while i < n:
                if not self._can_be_in_compound(morphemes_a[i], "start"):
                    i += 1
                    continue
                run: list[Morpheme] = [morphemes_a[i]]
                j = i + 1
                while j < n and self._can_be_in_compound(morphemes_a[j], "middle"):
                    run.append(morphemes_a[j])
                    j += 1
                # 末尾が接頭辞で終わる場合は削る
                while run and run[-1].part_of_speech()[0] == "接頭辞":
                    run.pop()

                if len(run) >= 2:
                    surf = "".join(m.surface() for m in run).strip()
                    norm = "".join(m.normalized_form() for m in run).strip()
                    yomi = "".join(m.reading_form() for m in run).strip()
                    any_oov = any(m.is_oov() for m in run)
                    register_candidate(
                        surface=surf,
                        normalized=norm,
                        yomi=yomi,
                        pos_str="複合名詞",
                        is_oov=any_oov,
                        is_compound=True,
                        sentence=sentence,
                    )
                i = max(i + 1, j)

        # 優先順位: OOVを優先し、次に出現回数順・文字数順でソート
        results = list(candidates_map.values())
        results.sort(key=lambda c: (not c.is_oov, -c.count, -len(c.surface)))
        return results
