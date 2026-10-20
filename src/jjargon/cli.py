import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from jjargon.analyzer import SudachiAnalyzer
from jjargon.extractor import (
    ExtractionMode,
    JargonExtractor,
    LLMConfig,
    SystemOneConfig,
)
from jjargon.models import TermStatus
from jjargon.reviewer import interactive_review
from jjargon.storage import load_glossary, save_glossary
from jjargon.sudachi_dict import export_sudachi_csv

load_dotenv()

app = typer.Typer(
    name="jjargon",
    help="日本語テキストからSudachiとAI(LLM)を用いて専門用語・誤認識候補を抽出・管理するCLIツール",
    no_args_is_help=True,
)
console = Console()


@app.command("analyze")
def analyze_cmd(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="解析対象の日本語テキストファイルパス", exists=True, readable=True
        ),
    ],
    glossary_path: Annotated[
        Path | None,
        typer.Option("--glossary", "-g", help="既存の専門用語集ファイル (YAML/JSON)"),
    ] = None,
    no_compounds: Annotated[
        bool,
        typer.Option(
            "--no-compounds", help="複合名詞の抽出を無効化し、is_oov()のみを抽出する"
        ),
    ] = False,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="表示する最大件数"),
    ] = 50,
) -> None:
    """Step 1 & 2 のみを実行: SudachiによるOOV（辞書未登録語）および複合名詞候補を抽出して表示します。"""
    text = input_file.read_text(encoding="utf-8")
    glossary = load_glossary(glossary_path) if glossary_path else None

    analyzer = SudachiAnalyzer()
    candidates = analyzer.analyze(
        text=text,
        glossary=glossary,
        include_compounds=not no_compounds,
    )

    if not candidates:
        console.print(
            "[yellow]候補となるOOVまたは複合名詞は見つかりませんでした。[/yellow]"
        )
        return

    table = Table(
        title=f"Sudachi 抽出候補一覧 (全 {len(candidates)} 件中 上位 {min(limit, len(candidates))} 件)"
    )
    table.add_column("表層形 (surface)", style="bold yellow", no_wrap=True)
    table.add_column("読み (yomi)", style="cyan", overflow="fold")
    table.add_column("種別", style="green")
    table.add_column("品詞", style="white")
    table.add_column("回数", justify="right")
    table.add_column("出現例 (抜粋)", style="dim", max_width=50)

    for c in candidates[:limit]:
        kind_badges = []
        if c.is_oov:
            kind_badges.append("[bold red]OOV[/bold red]")
        if c.is_compound:
            kind_badges.append("[blue]複合名詞[/blue]")
        example_str = c.examples[0] if c.examples else ""
        table.add_row(
            c.surface,
            c.yomi,
            " / ".join(kind_badges),
            c.pos,
            str(c.count),
            example_str,
        )

    console.print(table)


@app.command("extract")
def extract_cmd(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="抽出対象の日本語テキストファイルパス", exists=True, readable=True
        ),
    ],
    glossary_path: Annotated[
        Path,
        typer.Option("--glossary", "-g", help="保存先の専門用語集ファイル (YAML/JSON)"),
    ] = Path("glossary.yaml"),
    mode: Annotated[
        ExtractionMode,
        typer.Option(
            "--mode",
            "-M",
            help="専門用語抽出の実行モード (hybrid: System1トリアージ+System2生成, system-one-only: System1のみ, system-two-only: System2のみ)",
            case_sensitive=False,
        ),
    ] = ExtractionMode.HYBRID,
    domain: Annotated[
        str | None,
        typer.Option("--domain", "-d", help="部署名やプロジェクト名などのドメイン指定"),
    ] = None,
    # System Two (生成LLM) オプション
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="System Two (生成LLM) モデル名 (例: gpt-4o-mini, qwen2.5:14b)",
        ),
    ] = None,
    api_base: Annotated[
        str | None,
        typer.Option(
            "--api-base",
            help="System Two (生成LLM) API Base URL (環境変数 OPENAI_BASE_URL も利用可能)",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help="System Two (生成LLM) API Key (環境変数 OPENAI_API_KEY も利用可能)",
        ),
    ] = None,
    compact: Annotated[
        bool,
        typer.Option(
            "--compact/--full-schema",
            help="System Two の出力スキーマを極小化し、読み(yomi)・例文(examples)をSudachiから自動補完して高速化する",
        ),
    ] = True,
    # System One (意思決定モデル) オプション
    systemone_model: Annotated[
        str | None,
        typer.Option(
            "--systemone-model",
            help="System One モデル名 (例: kev:latest / 環境変数 SYSTEMONE_MODEL も利用可能)",
        ),
    ] = None,
    systemone_url: Annotated[
        str | None,
        typer.Option(
            "--systemone-url",
            help="System One API URL (例: http://localhost:11434/v1/systemone / 環境変数 SYSTEMONE_URL も利用可能)",
        ),
    ] = None,
    systemone_api_key: Annotated[
        str | None,
        typer.Option(
            "--systemone-api-key",
            "--systemone-key",
            help="System One API Key (環境変数 SYSTEMONE_API_KEY も利用可能)",
        ),
    ] = None,
    systemone_threshold: Annotated[
        float,
        typer.Option(
            "--systemone-threshold",
            help="System One の noul (専門用語らしさ) 採用閾値 (0.0〜1.0)",
        ),
    ] = 0.75,
    no_compounds: Annotated[
        bool,
        typer.Option(
            "--no-compounds", help="複合名詞の抽出を無効化し、OOVと文脈のみで抽出する"
        ),
    ] = False,
    review_now: Annotated[
        bool,
        typer.Option(
            "--review", "-r", help="抽出完了後にそのまま対話レビュー(Step 4)を開始する"
        ),
    ] = False,
) -> None:
    """Step 1〜3 を実行: Sudachiで候補を抽出し、AIで既存用語集と照合して新規候補・誤認識候補を保存します。"""
    text = input_file.read_text(encoding="utf-8")
    glossary = load_glossary(glossary_path)

    try:
        llm_config = LLMConfig.from_env(
            model=model,
            api_base=api_base,
            api_key=api_key,
            compact_mode=compact,
        )
        has_s1_options = bool(
            systemone_model
            or systemone_url
            or systemone_api_key
            or os.environ.get("SYSTEMONE_MODEL")
            or os.environ.get("SYSTEMONE_URL")
        )
        s1_config = (
            SystemOneConfig.from_env(
                model=systemone_model,
                url=systemone_url,
                api_key=systemone_api_key,
                threshold=systemone_threshold,
            )
            if (
                has_s1_options
                and ExtractionMode.from_str(mode) != ExtractionMode.SYSTEM_TWO_ONLY
            )
            or ExtractionMode.from_str(mode) == ExtractionMode.SYSTEM_ONE_ONLY
            else None
        )

        extractor = JargonExtractor(
            mode=mode,
            llm_config=llm_config,
            system_one_config=s1_config,
        )
    except ValueError as e:
        console.print(f"[red]エラー: {e}[/red]")
        raise typer.Exit(code=1) from e

    import time

    start_time = time.time()
    s1_info = (
        f" | System 1: [bold magenta]{extractor.system_one.model}[/bold magenta]"
        + (f" (th={extractor.system_one.threshold})" if extractor.system_one else "")
        if extractor.system_one
        else " | System 1: なし"
    )
    s2_info = (
        f"[bold]{extractor.system_two.model}[/bold] ({'compact' if compact else 'full'})"
        if extractor.system_two
        else "なし"
    )
    console.print(
        f"[bold cyan]1. テキスト読み込み完了[/bold cyan] ({len(text)} 文字) | "
        f"既存用語数: {len(glossary.terms)} 件 | モード: [bold green]{extractor.mode}[/bold green] | "
        f"LLM (System 2): {s2_info}{s1_info}"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("チャンク"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(
            "[bold green]Sudachi解析 & 専門用語抽出中[/bold green]",
            total=None,
        )
        _proposals, summary = extractor.extract_and_merge(
            text=text,
            glossary=glossary,
            domain=domain,
            include_compounds=not no_compounds,
            on_progress=lambda current, total: progress.update(
                task_id, completed=current, total=total
            ),
        )

    elapsed_time = time.time() - start_time
    save_glossary(glossary, glossary_path)

    stats = extractor.last_stats
    console.print(
        f"\n[bold green]抽出完了！[/bold green] (所要時間: {elapsed_time:.2f}秒) "
        f"用語集を保存しました: [underline]{glossary_path}[/underline]"
    )
    if extractor.system_one is not None:
        console.print(
            f"  [dim]📊 実行統計: チャンク数={stats['total_chunks']} (スキップ={stats['skipped_chunks']}) | "
            f"Sudachi候補={stats['candidates_total']}→System 1通過={stats['candidates_filtered']}件 | "
            f"System 1呼出={stats['systemone_calls']}回 | System 2 (LLM)呼出={stats['llm_calls']}回[/dim]"
        )
    else:
        console.print(
            f"  [dim]📊 実行統計: チャンク数={stats['total_chunks']} | "
            f"Sudachi候補={stats['candidates_total']}件 | System 2 (LLM)呼出={stats['llm_calls']}回[/dim]"
        )
    console.print(
        f"  • 新規候補の追加 (status: pending): [bold yellow]{len(summary['added'])}[/bold yellow] 件\n"
        f"  • 既存承認済み用語への追記提案 (pending_updates): [bold green]{len(summary['updated_pending'])}[/bold green] 件\n"
        f"  • 既存pending用語への統合: [bold cyan]{len(summary['merged_pending'])}[/bold cyan] 件\n"
        f"  • 却下済み用語としてスキップ: [dim]{len(summary['skipped_denied'])}[/dim] 件"
    )

    if summary["added"] or summary["updated_pending"] or summary["merged_pending"]:
        has_recognized = any(t.recognized_texts for t in summary["added"]) or any(
            t.pending_updates and t.pending_updates.recognized_texts
            for t in summary["updated_pending"]
        )
        table = Table(title="今回追加・更新された専門用語候補")
        table.add_column("種別", style="cyan")
        table.add_column("正式表記 (canonical_text)", style="bold yellow")
        table.add_column("読み (yomi)")
        table.add_column("別表記・略称 (aliases)")
        if has_recognized:
            table.add_column("誤認識 (recognized_texts)", style="magenta")
        table.add_column("意味 (meaning)", max_width=40)

        for t in summary["added"]:
            row = [
                "新規 (pending)",
                t.canonical_text,
                t.yomi,
                ", ".join(t.aliases),
            ]
            if has_recognized:
                row.append(", ".join(t.recognized_texts))
            row.append(t.meaning)
            table.add_row(*row)
        for t in summary["updated_pending"]:
            pu = t.pending_updates
            row = [
                "既存追記 (update)",
                t.canonical_text,
                t.yomi,
                ", ".join(pu.aliases if pu else []),
            ]
            if has_recognized:
                row.append(", ".join(pu.recognized_texts if pu else []))
            row.append(t.meaning)
            table.add_row(*row)
        console.print(table)
        console.print(
            f"\n[bold]次のステップ:[/bold] `jjargon review -g {glossary_path}` を実行して承認・却下を判断してください。"
        )

    if review_now:
        interactive_review(
            glossary=glossary,
            console=console,
            on_step_save=lambda g: save_glossary(g, glossary_path),
        )
        save_glossary(glossary, glossary_path)


@app.command("review")
def review_cmd(
    glossary_path: Annotated[
        Path,
        typer.Option("--glossary", "-g", help="専門用語集ファイル (YAML/JSON)"),
    ] = Path("glossary.yaml"),
) -> None:
    """Step 4 を実行: pending状態の専門用語候補や追記提案を人間が対話的に承認・編集・却下します。"""
    if not glossary_path.exists():
        console.print(f"[red]用語集ファイルが見つかりません: {glossary_path}[/red]")
        raise typer.Exit(code=1)

    glossary = load_glossary(glossary_path)
    stats = interactive_review(
        glossary=glossary,
        console=console,
        on_step_save=lambda g: save_glossary(g, glossary_path),
    )
    save_glossary(glossary, glossary_path)

    console.print(
        f"\n[bold]レビュー結果:[/bold] "
        f"承認: [green]{stats['approved']}[/green] 件 / "
        f"却下: [red]{stats['denied']}[/red] 件 / "
        f"スキップ: [yellow]{stats['skipped']}[/yellow] 件"
    )


@app.command("list")
def list_cmd(
    glossary_path: Annotated[
        Path,
        typer.Option("--glossary", "-g", help="専門用語集ファイル (YAML/JSON)"),
    ] = Path("glossary.yaml"),
    status: Annotated[
        TermStatus | None,
        typer.Option(
            "--status", "-s", help="ステータスで絞り込み (approved / pending / denied)"
        ),
    ] = None,
    domain: Annotated[
        str | None,
        typer.Option("--domain", "-d", help="ドメインで絞り込み"),
    ] = None,
) -> None:
    """専門用語集に登録されている用語の一覧を表示します。"""
    if not glossary_path.exists():
        console.print(
            f"[yellow]用語集ファイルがまだ存在しません: {glossary_path}[/yellow]"
        )
        return

    glossary = load_glossary(glossary_path)
    terms = glossary.terms
    if status is not None:
        terms = [t for t in terms if t.status == status]
    if domain is not None:
        terms = [t for t in terms if t.domain == domain]

    if not terms:
        console.print("[yellow]該当する用語はありません。[/yellow]")
        return

    table = Table(title=f"専門用語集一覧 ({glossary_path}) - {len(terms)} 件")
    table.add_column("Status", style="bold")
    table.add_column("正式表記", style="bold yellow")
    table.add_column("読み", style="cyan")
    table.add_column("別表記 (aliases)")
    table.add_column("誤認識 (recognized)", style="magenta")
    table.add_column("種類")
    table.add_column("ドメイン")
    table.add_column("意味", max_width=40)

    status_style = {
        TermStatus.APPROVED: "[green]approved[/green]",
        TermStatus.PENDING: "[yellow]pending[/yellow]",
        TermStatus.DENIED: "[red]denied[/red]",
    }

    for t in terms:
        st_label = status_style.get(t.status, str(t.status))
        if (
            t.status == TermStatus.APPROVED
            and t.pending_updates
            and t.pending_updates.has_items()
        ):
            st_label += "\n[bold green](+update)[/bold green]"
        table.add_row(
            st_label,
            t.canonical_text,
            t.yomi,
            ", ".join(t.aliases),
            ", ".join(t.recognized_texts),
            t.term_type,
            t.domain,
            t.meaning,
        )

    console.print(table)


@app.command("export-sudachi")
def export_sudachi_cmd(
    glossary_path: Annotated[
        Path,
        typer.Option("--glossary", "-g", help="専門用語集ファイル (YAML/JSON)"),
    ] = Path("glossary.yaml"),
    output_path: Annotated[
        Path,
        typer.Option("--output", "-o", help="出力するSudachiユーザー辞書CSVパス"),
    ] = Path("sudachi_user_dict.csv"),
    include_aliases: Annotated[
        bool,
        typer.Option(
            "--include-aliases/--no-aliases", help="aliasesも辞書エントリに含めるか"
        ),
    ] = True,
) -> None:
    """承認済み (status: approved) の専門用語からSudachiユーザー辞書用CSVファイルを出力します。"""
    if not glossary_path.exists():
        console.print(f"[red]用語集ファイルが見つかりません: {glossary_path}[/red]")
        raise typer.Exit(code=1)

    glossary = load_glossary(glossary_path)
    count = export_sudachi_csv(
        glossary=glossary,
        output_path=output_path,
        include_aliases=include_aliases,
    )
    console.print(
        f"[bold green]Sudachiユーザー辞書CSVを出力しました:[/bold green] {output_path} ({count} エントリ)"
    )
