# Discord bot ASR server

Klein の voice chat 機能向けに、日本語対応の Parakeet WebSocket ASR サーバーを提供します。NVIDIA NeMo の [`parakeet-tdt_ctc-0.6b-ja`](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) を使い、`/v1/asr` で 16 kHz・モノラル・signed 16-bit PCM を受け付けます。モデルは CC BY 4.0 で公開されており、初回起動時に Hugging Face からダウンロードされます。

## 起動

Linux 環境で動作します。Nix の開発シェルに含まれる Python 3.12 を使って、リポジトリのルートで起動します。

```sh
nix develop
uv sync
uv run python -m discord_bot_asr
```

`uv run asr-server` または `uv run parakeet-asr-server` でも起動できます。既定の接続先は `ws://localhost:8000/v1/asr` です。TLS 終端を設ける場合は `wss://` を使用してください。

NVIDIA GPU が利用できる場合、既定の `ASR_DEVICE=auto` は GPU を選びます。CPU で実行する場合は次のようにします。

```sh
ASR_DEVICE=cpu uv run python -m discord_bot_asr
```

## 設定

| 環境変数 | 既定値 | 説明 |
| --- | --- | --- |
| `ASR_DEVICE` | `auto` | 推論デバイス。`auto`、`cpu`、`cuda` を指定できます |
| `HOST` | `0.0.0.0` | bind するアドレス |
| `PORT` | `8000` | listen するポート |
| `LOG_LEVEL` | `INFO` | ログレベル |

各要求の音声は最大30秒です。サーバーは `requestId` ごとに完了または失敗を返し、応答待ちの間に次の音声をアップロードできます。

ユーザー systemd で常時起動・異常終了後の再起動を行う場合は、[`deploy/systemd/parakeet-asr.service.example`](deploy/systemd/parakeet-asr.service.example) を `~/.config/systemd/user/parakeet-asr.service` にコピーし、`/path/to/...` を実際のパスに置き換えてください。

```sh
systemctl --user daemon-reload
systemctl --user enable --now parakeet-asr.service
systemctl --user status parakeet-asr.service
```

ログアウト中も動かすには、ユーザーの linger を有効にしてください。Klein 側では voice chat を有効にし、`asrServerUrl` にサーバーの WebSocket URL を設定してください。プロトコルの詳細は [Voice chat ASR streaming protocol](https://github.com/OJII3/klein/blob/main/docs%2Fvoice-chat-asr-protocol.md) を参照してください。
