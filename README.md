# Discord bot ASR server

Klein の voice chat 機能向けに、Qwen3-ASR の WebSocket ストリーミングサーバーを提供します。Qwen3-ASR-0.6B を GPU 上で実行し、`/v1/stream` で 16 kHz・モノラル・signed 16-bit PCM を受け付けます。

## 起動

Linux と NVIDIA GPU が必要です。Nix の開発シェルに含まれる Python 3.12 と C コンパイラを使って、リポジトリのルートで起動します。

```sh
nix develop
uv sync
uv run qwen3-asr-server
```

初回起動時に Hugging Face からモデルをダウンロードします。既定の接続先は `ws://localhost:8000/v1/stream` です。TLS 終端を設ける場合は `wss://` を使用してください。

## 設定

| 環境変数 | 既定値 | 説明 |
| --- | --- | --- |
| `ASR_MODEL` | `Qwen/Qwen3-ASR-0.6B` | Qwen3-ASR モデル ID またはローカルパス |
| `GPU_MEMORY_UTILIZATION` | `0.55` | vLLM インスタンス用の GPU メモリ枠。RTX 3060 12 GB で SBV2 と共有する場合の初期値 |
| `MAX_MODEL_LEN` | `8192` | vLLM の最大シーケンス長。12 GB GPU で KV cache を抑える |
| `HOST` | `0.0.0.0` | bind するアドレス |
| `PORT` | `8000` | listen するポート |
| `LOG_LEVEL` | `INFO` | ログレベル |

`GPU_MEMORY_UTILIZATION` は vLLM インスタンスごとの制限で、他の GPU プロセスのメモリを管理するものではありません。0.55 は約 6.6 GiB を ASR 用に見込む設定ですが、SBV2 の推論時ピークや CUDA の一時メモリ次第で OOM は起こり得ます。実際の空き容量を `nvidia-smi` で確認し、必要なら値を下げてください。

長い発話はサーバー内部で30秒ごとに処理窓を切り替え、境界の音声を2秒重ねて認識文をつなぎます。クライアントからは1つの utterance のままで、`transcript.partial` は累積文、`transcript.final` は発話全体の文を返します。これにより発話全体の長さで Qwen の入力が増え続けないようにします。`MAX_MODEL_LEN` は各処理窓に対する上限です。

ユーザー systemd で常時起動・異常終了後の再起動を行う場合は、[`deploy/systemd/qwen3-asr.service.example`](deploy/systemd/qwen3-asr.service.example) を `~/.config/systemd/user/qwen3-asr.service` にコピーし、`/path/to/...` を実際のパスに置き換えてください。

```sh
systemctl --user daemon-reload
systemctl --user enable --now qwen3-asr.service
systemctl --user status qwen3-asr.service
```

ログアウト中も動かすには、ユーザーの linger を有効にしてください。Klein 側では voice chat を有効にし、`asrServerUrl` にサーバーの WebSocket URL を設定してください。プロトコルの詳細は [Voice chat ASR streaming protocol](https://github.com/OJII3/klein/blob/main/docs%2Fvoice-chat-asr-protocol.md) を参照してください。
