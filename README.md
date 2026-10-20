# jjargon

日本語テキスト（仕様書、社内Wiki、技術ドキュメント、業務マニュアル等）から、Sudachi形態素解析（未知語OOV抽出＋複合名詞抽出）とLLMを組み合わせて**一般的な辞書に載っていない専門用語・社内用語・略称・別表記**を抽出し、人間が対話的にレビュー・管理するためのCLIツールです。

## 特徴・処理フロー（フロー1: 専門用語登録）

1. **Step 1 & 2: Sudachiによる形態素解析と候補抽出 (`SudachiAnalyzer`)**
   - `SplitMode.A` と `SplitMode.C` を併用し、辞書未登録語（`is_oov() == True`）だけでなく、**連続する名詞を連結した複合名詞**も出現文脈（`examples`）付きで抽出します。
   - 既に `status: denied`（却下）として記録された不要語や記号・数値ノイズは事前に除外します。
2. **Step 3: AI（OpenAI API 互換 LLM）による用語集照合・候補作成 (`JargonExtractor`)**
   - 抽出されたOOV・複合名詞のヒントと元のテキスト文脈、および現在の専門用語集をLLMに渡し、OpenAI公式SDKの厳格な構造化出力（Strict Structured Outputs: Pydanticモデル）で以下を生成します：
     - **新規専門用語候補 (`new_term`)**: 正式表記 (`canonical_text`)、カタカナ読み (`yomi`)、別表記・英語略称 (`aliases`)、定義・意味 (`meaning`)、カテゴリ (`term_type`) を付与し `status: pending` として用語集に追加
     - **既存用語への別表記・略称の追加提案 (`update_existing`)**: 承認済み（`status: approved`）用語のステータスを維持したまま、`pending_updates.aliases` に差分として記録
3. **Step 4: 人間による対話的レビュー (`jjargon review`)**
   - CLI上で1件ずつ文脈（`examples`）を確認しながら、`Approve`（承認）/ `Edit`（インライン編集して承認）/ `Deny`（却下・以降の再抽出抑止）/ `Skip`（保留）を判断できます。
4. **Sudachiユーザー辞書CSVのエクスポート (`jjargon export-sudachi`)**
   - `approved` になった専門用語（および `aliases`）から Sudachi ユーザー辞書ビルド用の18カラムCSVを出力できます。

---

## セットアップ

```bash
uv sync
```

### LLMの設定（OpenAI API 互換サーバー / クラウド）

OpenAI API 互換のエンドポイント（本家 OpenAI, vLLM, Ollama `/v1`, LM Studio, llama.cpp 等）をサポートしています。
一般的な環境変数（`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`）または CLI オプションで指定します。

```bash
# OpenAI公式を使う場合 (デフォルトモデル: gpt-4o-mini)
export OPENAI_API_KEY="sk-..."

# 互換プロバイダ（Groq, DeepSeek, OpenRouter等）やローカル推論サーバーを使う場合
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_MODEL="qwen2.5:14b"

# System One モデル（Ollama /v1/systemone, kev:latest 等）を併用する場合
export SYSTEMONE_URL="http://localhost:11434/v1/systemone"
export SYSTEMONE_MODEL="kev:latest"
# 認証が必要なエンドポイント（クラウドやプロキシ経由など）を利用する場合
export SYSTEMONE_API_KEY="your-systemone-api-key"
```

---

## 使い方 (CLI)

### 1. SudachiによるOOV・複合名詞抽出のみを確認する（AI呼び出しなし）

```bash
uv run jjargon analyze sample.txt
```

### 2. テキストから専門用語・誤認識候補を抽出して用語集に保存する（Step 1〜3）

```bash
# ハイブリッドモード (デフォルト): System One (kev:latest) で事前トリアージし、System Two (生成LLM) で構造化抽出
uv run jjargon extract sample.txt --glossary glossary.yaml --domain "情報システム部"
# または明示指定:
uv run jjargon extract sample.txt -g glossary.yaml --mode hybrid

# System One 完結モード: 生成LLMを一切使わず、Sudachi + System One のみで超高速抽出
uv run jjargon extract sample.txt -g glossary.yaml --mode system-one-only

# System Two 単体モード: System One を使わず、生成LLM単体で抽出
uv run jjargon extract sample.txt -g glossary.yaml --mode system-two-only

# 従来の全フィールドLLM出力スキーマを使う場合 (--compact がデフォルト)
uv run jjargon extract sample.txt -g glossary.yaml --full-schema
```

### 3. 抽出された候補を人間がレビューして承認・編集・却下する（Step 4）

```bash
uv run jjargon review --glossary glossary.yaml
```

- `[a]` Approve: 承認（`status: approved` に変更、または `pending_updates` をマージ）
- `[e]` Edit & Approve: 表記・読み・別表記・誤認識・意味・カテゴリを対話編集して承認
- `[d]` Deny: 却下（新規候補は `status: denied` として記録し次回以降の再抽出を防止、既存用語の `pending_updates` は破棄）
- `[s]` Skip: スキップ
- `[q]` Quit: 中断して保存

### 4. 用語集の一覧表示

```bash
uv run jjargon list --glossary glossary.yaml
uv run jjargon list --glossary glossary.yaml --status approved
```

### 5. Sudachiユーザー辞書用CSVのエクスポート

```bash
uv run jjargon export-sudachi --glossary glossary.yaml --output sudachi_user_dict.csv
```
