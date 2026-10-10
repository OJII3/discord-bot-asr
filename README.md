# Discord bot ASR server

Klein の voice chat 機能向けに、faster-whisper の WebSocket ASR サーバーを提供します。既定で `large-v3-turbo` を実行し、`/v1/asr` で 16 kHz・モノラル・signed 16-bit PCM を受け付けます。

## 起動

Linux 環境で動作します。Nix の開発シェルに含まれる Python 3.12 と C コンパイラを使って、リポジトリのルートで起動します。NVIDIA GPU がある場合は自動で利用します。

```sh
nix develop
uv sync --extra cuda
uv run asr-server
```

NVIDIA GPU を使う場合、`cuda` extra が CUDA 12 用 cuBLAS と cuDNN 9 をインストールし、Nix 開発シェルがそれらのライブラリをロード対象にします。初回起動時に Hugging Face からモデルをダウンロードします。既定の接続先は `ws://localhost:8000/v1/asr` です。TLS 終端を設ける場合は `wss://` を使用してください。

GPU ライブラリを入れず CPU で実行する場合は `uv sync` の後、次のように起動します。

```sh
ASR_DEVICE=cpu uv run asr-server
```

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

## Parakeet サーバー

既存 faster-whisper サーバーと同じ `/v1/asr` WebSocket プロトコルを使う Parakeet サーバーも起動できます。NVIDIA NeMo の `parakeet-tdt-0.6b-v3` を使用します。Parakeet v3 は英語を含む欧州25言語に対応しますが、日本語には対応していません。Klein の `language` を対応言語（例: `en`）にして試してください。

```sh
nix develop
uv sync --extra parakeet
uv run parakeet-asr-server
```

NeMo/PyTorch のインストールは faster-whisper 用 `cuda` extra と別です。GPU 推論には CUDA 対応 PyTorch 環境が必要です。`ASR_DEVICE=cpu` または `ASR_DEVICE=cuda` でデバイスを指定でき、既定の `auto` は CUDA が利用可能なら GPU を選びます。Parakeet サーバーは既定で `ws://localhost:8000/v1/asr` を listen するため、faster-whisper サーバーと同時には同じポートで起動できません。

ユーザー systemd で常時起動・異常終了後の再起動を行う場合は、[`deploy/systemd/faster-whisper-asr.service.example`](deploy/systemd/faster-whisper-asr.service.example) を `~/.config/systemd/user/faster-whisper-asr.service` にコピーし、`/path/to/...` を実際のパスに置き換えてください。

```sh
systemctl --user daemon-reload
systemctl --user enable --now faster-whisper-asr.service
systemctl --user status faster-whisper-asr.service
```

ログアウト中も動かすには、ユーザーの linger を有効にしてください。Klein 側では voice chat を有効にし、`asrServerUrl` にサーバーの WebSocket URL を設定してください。プロトコルの詳細は [Voice chat ASR streaming protocol](https://github.com/OJII3/klein/blob/main/docs%2Fvoice-chat-asr-protocol.md) を参照してください。
