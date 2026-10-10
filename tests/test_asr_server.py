from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import asr_server


class FakeWhisperModel:
    supported_languages = ["ja", "en"]

    def __init__(self) -> None:
        self.transcribed_audio: list[np.ndarray] = []

    def transcribe(self, audio: np.ndarray, **_: object):
        self.transcribed_audio.append(audio)
        return [SimpleNamespace(text=" test")], None


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    model = FakeWhisperModel()

    @asynccontextmanager
    async def skip_model_loading(_):
        yield

    monkeypatch.setattr(asr_server, "asr", model)
    monkeypatch.setattr(asr_server.app.router, "lifespan_context", skip_model_loading)
    with TestClient(asr_server.app) as test_client:
        yield test_client, model


def test_completes_request_with_normalized_pcm(client) -> None:
    test_client, model = client
    with test_client.websocket_connect("/v1/asr") as websocket:
        websocket.send_json({"type": "asr.start", "requestId": "one"})
        websocket.send_bytes(np.array([32767, -32768], dtype="<i2").tobytes())
        websocket.send_json({"type": "asr.commit", "requestId": "one"})

        assert websocket.receive_json() == {
            "type": "asr.completed",
            "requestId": "one",
            "text": "test",
        }

    np.testing.assert_allclose(model.transcribed_audio[0], [32767 / 32768, -1.0])


def test_allows_next_upload_while_prior_result_is_pending(client) -> None:
    test_client, _ = client
    with test_client.websocket_connect("/v1/asr") as websocket:
        for request_id in ("one", "two"):
            websocket.send_json({"type": "asr.start", "requestId": request_id})
            websocket.send_bytes(np.array([1], dtype="<i2").tobytes())
            websocket.send_json({"type": "asr.commit", "requestId": request_id})

        responses = [websocket.receive_json(), websocket.receive_json()]

    assert {response["requestId"] for response in responses} == {"one", "two"}
    assert all(response["type"] == "asr.completed" for response in responses)


def test_rejects_audio_over_30_seconds(client) -> None:
    test_client, model = client
    too_long_pcm = np.zeros(16_000 * 30 + 1, dtype="<i2").tobytes()
    with test_client.websocket_connect("/v1/asr") as websocket:
        websocket.send_json({"type": "asr.start", "requestId": "long"})
        websocket.send_bytes(too_long_pcm)
        websocket.send_json({"type": "asr.commit", "requestId": "long"})

        assert websocket.receive_json() == {
            "type": "asr.failed",
            "requestId": "long",
            "code": "audio_too_long",
            "message": "Audio uploads must not exceed 30 seconds.",
        }

    assert model.transcribed_audio == []
