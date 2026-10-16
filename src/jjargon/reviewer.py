import sys
from collections.abc import Callable

from prompt_toolkit import prompt as pt_prompt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from jjargon.models import Glossary, Term, TermStatus, utc_now


def _default_input(message: str, default: str = "") -> str:
    """TTY環境ではprompt_toolkitでデフォルト値付き入力を行い、非TTY（テスト等）では標準入力を使う"""
    if sys.stdin.isatty():
        return pt_prompt(message, default=default).strip()
    sys.stdout.write(f"{message}[{default}]: ")
    sys.stdout.flush()
    line = sys.stdin.readline()
    if not line:
        return default
    val = line.strip()
    return val if val else default


def _split_csv(text: str) -> list[str]:
    if not text.strip():
        return []
    # カンマまたは読点（、）で分割
    parts = [p.strip() for p in text.replace("、", ",").split(",")]
    return list(dict.fromkeys(p for p in parts if p))


def render_term_card(term: Term, index: int, total: int, console: Console) -> None:
    """レビュー対象のTermをRichのPanelとして見やすく表示する"""
    is_update = term.status == TermStatus.APPROVED and term.pending_updates is not None
    title_prefix = "【既存用語への追記提案】" if is_update else "【新規専門用語候補】"

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Field", style="bold cyan", width=20)
    table.add_column("Value", style="white")

    table.add_row("ID", term.id)
    table.add_row(
        "正式表記 (canonical)", f"[bold yellow]{term.canonical_text}[/bold yellow]"
    )
    table.add_row("読み (yomi)", term.yomi or "[dim](未設定)[/dim]")
    table.add_row("意味 (meaning)", term.meaning or "[dim](未設定)[/dim]")
    table.add_row("種類 (term_type)", term.term_type or "[dim](未設定)[/dim]")
    table.add_row("ドメイン (domain)", term.domain or "[dim](未設定)[/dim]")
    table.add_row(
        "別表記 (aliases)",
        ", ".join(term.aliases) if term.aliases else "[dim](なし)[/dim]",
    )
    table.add_row(
        "誤認識 (recognized)",
        ", ".join(term.recognized_texts)
        if term.recognized_texts
        else "[dim](なし)[/dim]",
    )

    if term.examples:
        ex_text = "\n".join(f"• {ex}" for ex in term.examples)
        table.add_row("出現例 (examples)", ex_text)

    if is_update and term.pending_updates:
        pu = term.pending_updates
        table.add_row("", "")
        table.add_row(
            "[bold green]＋ 追加別表記候補[/bold green]",
            f"[bold green]{', '.join(pu.aliases) or '(なし)'}[/bold green]",
        )
        table.add_row(
            "[bold green]＋ 追加誤認識候補[/bold green]",
            f"[bold green]{', '.join(pu.recognized_texts) or '(なし)'}[/bold green]",
        )
        if pu.examples:
            pu_ex = "\n".join(f"• {ex}" for ex in pu.examples)
            table.add_row("[bold green]＋ 追加出現例[/bold green]", pu_ex)
        if pu.reason:
            table.add_row("[bold green]AI提案理由[/bold green]", pu.reason)

    border_style = "green" if is_update else "yellow"
    panel = Panel(
        table,
        title=f"[{index}/{total}] {title_prefix} {term.canonical_text}",
        border_style=border_style,
    )
    console.print(panel)


def interactive_review(
    glossary: Glossary,
    console: Console | None = None,
    input_fn: Callable[[str, str], str] | None = None,
    on_step_save: Callable[[Glossary], None] | None = None,
) -> dict[str, int]:
    """
    未承認（pending または pending_updates あり）の用語を対話的にレビューする。
    戻り値: {'approved': N, 'denied': N, 'skipped': N}
    """
    console = console or Console()
    ask = input_fn or _default_input

    pending_terms = glossary.get_pending_reviews()
    stats = {"approved": 0, "denied": 0, "skipped": 0}

    if not pending_terms:
        console.print(
            "[bold green]レビュー待ち（pending）の用語はありません。[/bold green]"
        )
        return stats

    total = len(pending_terms)
    console.print(
        f"\n[bold]全 {total} 件のレビュー待ち項目があります。[/bold]\n"
        "操作キー: [bold green][a][/bold green]承認(Approve) / "
        "[bold cyan][e][/bold cyan]編集して承認(Edit) / "
        "[bold red][d][/bold red]却下(Deny) / "
        "[bold yellow][s][/bold yellow]スキップ(Skip) / "
        "[bold magenta][q][/bold magenta]中断して保存(Quit)\n"
    )

    for idx, term in enumerate(pending_terms, start=1):
        render_term_card(term, idx, total, console)
        is_update = (
            term.status == TermStatus.APPROVED and term.pending_updates is not None
        )

        while True:
            choice = ask("判断を選択してください (a/e/d/s/q): ", "a").lower()
            if choice in {"a", "e", "d", "s", "q"}:
                break
            console.print("[red]a, e, d, s, q のいずれかを入力してください。[/red]")

        if choice == "q":
            console.print(
                "[yellow]レビューを中断します。ここまでの変更を保存します。[/yellow]"
            )
            break

        if choice == "s":
            stats["skipped"] += 1
            console.print("[dim]スキップしました。[/dim]\n")
            continue

        if choice == "d":
            if is_update:
                term.discard_pending_updates()
                console.print(
                    f"[red]『{term.canonical_text}』への追記提案を却下しました（本体はapprovedのまま維持）。[/red]\n"
                )
            else:
                term.status = TermStatus.DENIED
                term.updated_at = utc_now()
                console.print(
                    f"[red]『{term.canonical_text}』を却下 (status: denied) に設定しました。[/red]\n"
                )
            stats["denied"] += 1
            if on_step_save:
                on_step_save(glossary)
            continue

        if choice == "e":
            if is_update and term.pending_updates:
                pu = term.pending_updates
                aliases_str = ask(
                    "追加する別表記 (カンマ区切り): ", ", ".join(pu.aliases)
                )
                rec_str = ask(
                    "追加する誤認識 (カンマ区切り): ", ", ".join(pu.recognized_texts)
                )
                pu.aliases = _split_csv(aliases_str)
                pu.recognized_texts = _split_csv(rec_str)
                term.apply_pending_updates()
            else:
                term.canonical_text = ask(
                    "正式表記 (canonical_text): ", term.canonical_text
                )
                term.yomi = ask("読み仮名 (yomi): ", term.yomi)
                aliases_str = ask(
                    "類義語・別表記 (カンマ区切り): ", ", ".join(term.aliases)
                )
                rec_str = ask(
                    "誤認識一覧 (カンマ区切り): ", ", ".join(term.recognized_texts)
                )
                term.meaning = ask("意味 (meaning): ", term.meaning)
                term.term_type = ask("種類 (term_type): ", term.term_type)
                term.domain = ask("ドメイン (domain): ", term.domain)
                term.aliases = _split_csv(aliases_str)
                term.recognized_texts = _split_csv(rec_str)
                term.status = TermStatus.APPROVED
                term.updated_at = utc_now()

            stats["approved"] += 1
            console.print(
                f"[green]『{term.canonical_text}』を編集して承認しました。[/green]\n"
            )
            if on_step_save:
                on_step_save(glossary)
            continue

        if choice == "a":
            if is_update:
                term.apply_pending_updates()
            else:
                term.status = TermStatus.APPROVED
                term.updated_at = utc_now()
            stats["approved"] += 1
            console.print(f"[green]『{term.canonical_text}』を承認しました。[/green]\n")
            if on_step_save:
                on_step_save(glossary)

    return stats
