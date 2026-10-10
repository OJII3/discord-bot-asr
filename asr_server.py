import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
from faster_whisper import WhisperModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 30
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_SECONDS

asr: WhisperModel | None = None
inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global asr
    model_name = os.getenv("ASR_MODEL", "large-v3-turbo")
    logger.info("Loading %s", model_name)
    asr = await asyncio.to_thread(
        WhisperModel,
        model_name,
        device=os.getenv("ASR_DEVICE", "auto"),
        compute_type=os.getenv("ASR_COMPUTE_TYPE", "default"),
    )
    logger.info("Model ready")
    yield
    asr = None


app = FastAPI(lifespan=lifespan)


def valid_object(value: object, keys: set[str]) -> bool:
    return isinstance(value, dict) and set(value) == keys


def transcribe(audio: np.ndarray, language: str) -> str:
    if asr is None:
        raise RuntimeError("ASR model is not ready")
    segments, _ = asr.transcribe(
        audio,
        language=language,
        task="transcribe",
        beam_size=5,
    )
    return "".join(segment.text for segment in segments).strip()


@app.websocket("/v1/asr")
async def asr_session(websocket: WebSocket) -> None:
    await websocket.accept()
    upload: dict[str, object] | None = None
    seen_request_ids: set[str] = set()
    send_lock = asyncio.Lock()
    inference_tasks: set[asyncio.Task[None]] = set()

    async def send(message: dict[str, str]) -> None:
        async with send_lock:
            await websocket.send_json(message)

    async def fail_protocol(message: str) -> None:
        logger.debug("Closing ASR connection after protocol error: %s", message)
        await websocket.close(code=1008)

    async def run_inference(request_id: str, language: str, pcm: bytes) -> None:
        try:
            supported_languages = asr.supported_languages if asr is not None else []
            if language not in supported_languages:
                await send({
                    "type": "asr.failed",
                    "requestId": request_id,
                    "code": "unsupported_language",
                    "message": "The requested language is not supported.",
                })
                return

            audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            async with inference_lock:
                text = await asyncio.to_thread(transcribe, audio, language)
            await send({"type": "asr.completed", "requestId": request_id, "text": text})
        except (WebSocketDisconnect, RuntimeError):
            logger.exception("ASR request %s failed", request_id)
            try:
                await send({
                    "type": "asr.failed",
                    "requestId": request_id,
                    "code": "inference_failed",
                    "message": "Speech recognition failed.",
                })
            except (RuntimeError, WebSocketDisconnect):
                pass
        except Exception:
            logger.exception("ASR request %s failed", request_id)
            try:
                await send({
                    "type": "asr.failed",
                    "requestId": request_id,
                    "code": "inference_failed",
                    "message": "Speech recognition failed.",
                })
            except (RuntimeError, WebSocketDisconnect):
                pass

    try:
        while True:
            frame = await websocket.receive()

            if frame.get("bytes") is not None:
                payload = frame["bytes"]
                if upload is None:
                    await fail_protocol("Audio arrived without an active request")
                    return
                if not payload or len(payload) % 2:
                    await fail_protocol("Audio frames must contain whole PCM samples")
                    return
                if not upload["too_long"]:
                    audio = upload["audio"]
                    if len(audio) + len(payload) > MAX_AUDIO_SAMPLES * 2:
                        upload["too_long"] = True
                        upload["audio"] = bytearray()
                    else:
                        audio.extend(payload)
                continue

            raw_text = frame.get("text")
            if raw_text is None:
                await fail_protocol("Expected a JSON text frame or binary audio frame")
                return
            try:
                message = json.loads(raw_text)
            except json.JSONDecodeError:
                await fail_protocol("Control frames must be JSON objects")
                return
            if not isinstance(message, dict):
                await fail_protocol("Control frames must be JSON objects")
                return

            message_type = message.get("type")
            if message_type == "asr.start":
                if (
                    (set(message) not in ({"type", "requestId"}, {"type", "requestId", "language"}))
                    or upload is not None
                ):
                    await fail_protocol("Unexpected or invalid asr.start frame")
                    return
                request_id = message["requestId"]
                language = message.get("language", "ja")
                if (
                    not isinstance(request_id, str)
                    or not request_id
                    or request_id in seen_request_ids
                    or not isinstance(language, str)
                    or not language
                ):
                    await fail_protocol("requestId and language must be valid strings")
                    return
                seen_request_ids.add(request_id)
                upload = {
                    "request_id": request_id,
                    "language": language,
                    "audio": bytearray(),
                    "too_long": False,
                }
                continue

            if message_type == "asr.commit":
                if (
                    not valid_object(message, {"type", "requestId"})
                    or upload is None
                    or message["requestId"] != upload["request_id"]
                ):
                    await fail_protocol("Unexpected or invalid asr.commit frame")
                    return

                request_id = upload["request_id"]
                if upload["too_long"]:
                    await send({
                        "type": "asr.failed",
                        "requestId": request_id,
                        "code": "audio_too_long",
                        "message": "Audio uploads must not exceed 30 seconds.",
                    })
                elif not upload["audio"]:
                    await send({
                        "type": "asr.failed",
                        "requestId": request_id,
                        "code": "empty_audio",
                        "message": "Audio upload must contain at least one PCM frame.",
                    })
                else:
                    task = asyncio.create_task(
                        run_inference(request_id, upload["language"], bytes(upload["audio"]))
                    )
                    inference_tasks.add(task)
                    task.add_done_callback(inference_tasks.discard)
                upload = None
                continue

            await fail_protocol("Unexpected control frame")
            return
    except WebSocketDisconnect:
        return


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
