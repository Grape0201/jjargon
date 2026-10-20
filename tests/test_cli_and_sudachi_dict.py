import csv
from pathlib import Path

from typer.testing import CliRunner

from jjargon.cli import app
from jjargon.models import Glossary, Term, TermStatus
from jjargon.storage import save_glossary

runner = CliRunner()


def test_cli_analyze_and_list_and_export_sudachi(tmp_path: Path) -> None:
    # 1. サンプルテキストファイル作成
    sample_txt = tmp_path / "meeting.txt"
    sample_txt.write_text(
        "本日はKintone連携基盤と稟議ワークフローの改修について話し合いました。\n",
        encoding="utf-8",
    )

    # 2. jjargon analyze の実行テスト
    res_analyze = runner.invoke(app, ["analyze", str(sample_txt)])
    assert res_analyze.exit_code == 0
    assert "稟議ワークフロー" in res_analyze.stdout

    # 3. サンプルGlossaryの保存と jjargon list の実行テスト
    glossary_file = tmp_path / "glossary.yaml"
    glossary = Glossary(
        terms=[
            Term(
                canonical_text="Kintone",
                yomi="キントーン",
                aliases=["kintone"],
                recognized_texts=["均等ん"],
                meaning="業務アプリ構築サービス",
                term_type="service",
                domain="情シス",
                status=TermStatus.APPROVED,
            ),
            Term(
                canonical_text="未承認用語",
                yomi="ミショウニンヨウゴ",
                status=TermStatus.PENDING,
            ),
        ]
    )
    save_glossary(glossary, glossary_file)

    res_list = runner.invoke(
        app, ["list", "-g", str(glossary_file), "--status", "approved"]
    )
    assert res_list.exit_code == 0
    assert "Kintone" in res_list.stdout
    assert "未承認用語" not in res_list.stdout

    # 4. jjargon export-sudachi の実行テスト（approvedのみが出力され、18カラムCSVになっていること）
    dict_csv = tmp_path / "user_dict.csv"
    res_export = runner.invoke(
        app,
        ["export-sudachi", "-g", str(glossary_file), "-o", str(dict_csv)],
    )
    assert res_export.exit_code == 0
    assert dict_csv.exists()

    rows = list(csv.reader(dict_csv.read_text(encoding="utf-8").splitlines()))
    # Kintone (canonical) と kintone (alias) の2行が出力され、未承認用語は出力されない
    assert len(rows) == 2
    assert len(rows[0]) == 18
    assert rows[0][0] == "Kintone"
    assert rows[0][11] == "キントーン"
    assert rows[0][12] == "Kintone"
    assert rows[1][0] == "kintone"
    assert rows[1][12] == "Kintone"  # aliasも正規化表記はKintoneに統一される


def test_cli_extract_mode_missing_systemone_error(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("SYSTEMONE_MODEL", raising=False)
    monkeypatch.delenv("SYSTEMONE_URL", raising=False)

    sample_txt = tmp_path / "sample.txt"
    sample_txt.write_text("テストテキスト", encoding="utf-8")

    # system-one-only モードで環境変数/引数なしの場合に終了コード1でエラーになること
    result = runner.invoke(app, ["extract", str(sample_txt), "--mode", "system-one-only"])
    assert result.exit_code == 1
    assert "System One モデルが指定されていません" in result.stdout

