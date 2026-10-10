from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import parakeet_asr_server


class FakeParakeetModel:
    def __init__(self) -> None:
        self.transcribed_audio: list[np.ndarray] = []

    def transcribe(self, audio: list[np.ndarray], **_: object):
        self.transcribed_audio.extend(audio)
        return [SimpleNamespace(text=" こんにちは。 ")]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    model = FakeParakeetModel()

    @asynccontextmanager
    async def skip_model_loading(_):
        yield

    monkeypatch.setattr(parakeet_asr_server, "asr", model)
    monkeypatch.setattr(parakeet_asr_server.app.router, "lifespan_context", skip_model_loading)
    with TestClient(parakeet_asr_server.app) as test_client:
        yield test_client, model


def test_completes_japanese_request_with_normalized_pcm(client) -> None:
    test_client, model = client
    with test_client.websocket_connect("/v1/asr") as websocket:
        websocket.send_json({"type": "asr.start", "requestId": "ja-one", "language": "ja"})
        websocket.send_bytes(np.array([32767, -32768], dtype="<i2").tobytes())
        websocket.send_json({"type": "asr.commit", "requestId": "ja-one"})

        assert websocket.receive_json() == {
            "type": "asr.completed",
            "requestId": "ja-one",
            "text": "こんにちは。",
        }

    np.testing.assert_allclose(model.transcribed_audio[0], [32767 / 32768, -1.0])


def test_rejects_unsupported_language_for_request_only(client) -> None:
    test_client, model = client
    with test_client.websocket_connect("/v1/asr") as websocket:
        websocket.send_json({"type": "asr.start", "requestId": "en-one", "language": "en"})
        websocket.send_bytes(np.array([1], dtype="<i2").tobytes())
        websocket.send_json({"type": "asr.commit", "requestId": "en-one"})

        assert websocket.receive_json() == {
            "type": "asr.failed",
            "requestId": "en-one",
            "code": "unsupported_language",
            "message": "The requested language is not supported.",
        }

    assert model.transcribed_audio == []


def test_rejects_audio_over_30_seconds(client) -> None:
    test_client, model = client
    too_long_pcm = np.zeros(16_000 * 30 + 1, dtype="<i2").tobytes()
    with test_client.websocket_connect("/v1/asr") as websocket:
        websocket.send_json({"type": "asr.start", "requestId": "long", "language": "ja"})
        websocket.send_bytes(too_long_pcm)
        websocket.send_json({"type": "asr.commit", "requestId": "long"})

        assert websocket.receive_json() == {
            "type": "asr.failed",
            "requestId": "long",
            "code": "audio_too_long",
            "message": "Audio uploads must not exceed 30 seconds.",
        }

    assert model.transcribed_audio == []
