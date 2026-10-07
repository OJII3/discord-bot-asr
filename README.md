# Discord bot ASR server

Klein の voice chat 機能向けに、Qwen3-ASR の WebSocket ストリーミングサーバーを提供します。Qwen3-ASR-0.6B を GPU 上で実行し、`/v1/stream` で 16 kHz・モノラル・signed 16-bit PCM を受け付けます。

## 起動

Linux と NVIDIA GPU、CUDA 対応の vLLM 実行環境が必要です。uv と Python 3.12 を用意して、リポジトリのルートで実行します。

```sh
uv sync
uv run python asr_server.py
```

初回起動時に Hugging Face からモデルをダウンロードします。既定の接続先は `ws://localhost:8000/v1/stream` です。TLS 終端を設ける場合は `wss://` を使用してください。

## 設定

| 環境変数 | 既定値 | 説明 |
| --- | --- | --- |
| `ASR_MODEL` | `Qwen/Qwen3-ASR-0.6B` | Qwen3-ASR モデル ID またはローカルパス |
| `GPU_MEMORY_UTILIZATION` | `0.8` | vLLM の GPU メモリ使用率 |
| `HOST` | `0.0.0.0` | bind するアドレス |
| `PORT` | `8000` | listen するポート |
| `LOG_LEVEL` | `INFO` | ログレベル |

Klein 側では voice chat を有効にし、`asrServerUrl` にサーバーの WebSocket URL を設定してください。プロトコルの詳細は [Voice chat ASR streaming protocol](https://github.com/OJII3/klein/blob/main/docs%2Fvoice-chat-asr-protocol.md) を参照してください。
