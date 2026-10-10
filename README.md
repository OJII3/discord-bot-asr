# Discord bot ASR server

Klein の voice chat 機能向けに、faster-whisper の WebSocket ASR サーバーを提供します。既定で `large-v3-turbo` を実行し、`/v1/asr` で 16 kHz・モノラル・signed 16-bit PCM を受け付けます。

## 起動

Linux 環境で動作します。Nix の開発シェルに含まれる Python 3.12 と C コンパイラを使って、リポジトリのルートで起動します。NVIDIA GPU がある場合は自動で利用します。

```sh
nix develop
uv sync
uv run asr-server
```

初回起動時に Hugging Face からモデルをダウンロードします。既定の接続先は `ws://localhost:8000/v1/asr` です。TLS 終端を設ける場合は `wss://` を使用してください。

## 設定

| 環境変数 | 既定値 | 説明 |
| --- | --- | --- |
| `ASR_MODEL` | `large-v3-turbo` | faster-whisper モデル名またはローカルパス |
| `ASR_DEVICE` | `auto` | faster-whisper の実行デバイス |
| `ASR_COMPUTE_TYPE` | `default` | CTranslate2 の計算型 |
| `HOST` | `0.0.0.0` | bind するアドレス |
| `PORT` | `8000` | listen するポート |
| `LOG_LEVEL` | `INFO` | ログレベル |

各要求の音声は最大30秒です。サーバーは `requestId` ごとに完了または失敗を返し、応答待ちの間に次の音声をアップロードできます。

ユーザー systemd で常時起動・異常終了後の再起動を行う場合は、[`deploy/systemd/faster-whisper-asr.service.example`](deploy/systemd/faster-whisper-asr.service.example) を `~/.config/systemd/user/faster-whisper-asr.service` にコピーし、`/path/to/...` を実際のパスに置き換えてください。

```sh
systemctl --user daemon-reload
systemctl --user enable --now faster-whisper-asr.service
systemctl --user status faster-whisper-asr.service
```

ログアウト中も動かすには、ユーザーの linger を有効にしてください。Klein 側では voice chat を有効にし、`asrServerUrl` にサーバーの WebSocket URL を設定してください。プロトコルの詳細は [Voice chat ASR streaming protocol](https://github.com/OJII3/klein/blob/main/docs%2Fvoice-chat-asr-protocol.md) を参照してください。
