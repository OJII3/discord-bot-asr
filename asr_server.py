import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from qwen_asr import Qwen3ASRModel


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
LANGUAGES = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "mk": "Macedonian",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "ro": "Romanian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}

asr: Qwen3ASRModel | None = None
inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global asr
    model_name = os.getenv("ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
    gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.8"))
    logger.info("Loading %s", model_name)
    asr = await asyncio.to_thread(
        Qwen3ASRModel.LLM,
        model=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        max_new_tokens=128,
    )
    logger.info("Model ready")
    yield
    asr = None


app = FastAPI(lifespan=lifespan)


async def send_error(websocket: WebSocket, code: str, message: str) -> None:
    await websocket.send_json({"type": "error", "code": code, "message": message})
    await websocket.close(code=1008)


def valid_object(value: object, keys: set[str]) -> bool:
    return isinstance(value, dict) and set(value) == keys


@app.websocket("/v1/stream")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session_ready = False
    utterance_id: str | None = None
    state = None
    session_language = LANGUAGES["ja"]
    last_partial_text = ""

    try:
        while True:
            frame = await websocket.receive()

            if frame.get("bytes") is not None:
                payload = frame["bytes"]
                if not session_ready or utterance_id is None:
                    await send_error(websocket, "invalid_message", "Audio is only allowed during an utterance.")
                    return
                if not payload or len(payload) % 2:
                    await send_error(websocket, "invalid_audio_frame", "Audio frames must contain whole PCM samples.")
                    return

                pcm = np.frombuffer(payload, dtype="<i2")
                async with inference_lock:
                    await asyncio.to_thread(asr.streaming_transcribe, pcm, state)
                    text = state.text
                if text != last_partial_text:
                    last_partial_text = text
                    await websocket.send_json({
                        "type": "transcript.partial",
                        "utteranceId": utterance_id,
                        "text": text,
                    })
                continue

            raw_text = frame.get("text")
            if raw_text is None:
                await send_error(websocket, "invalid_message", "Expected a JSON text frame or a binary audio frame.")
                return
            try:
                message = json.loads(raw_text)
            except json.JSONDecodeError:
                await send_error(websocket, "invalid_message", "Control frames must be JSON objects.")
                return

            if not isinstance(message, dict):
                await send_error(websocket, "invalid_message", "Control frames must be JSON objects.")
                return

            message_type = message.get("type")
            if not session_ready:
                if (
                    not isinstance(message, dict)
                    or set(message) - {"type", "language", "audio"}
                    or not {"type", "audio"}.issubset(message)
                    or message_type != "session.start"
                ):
                    await send_error(websocket, "invalid_message", "Expected a session.start frame.")
                    return
                language = message.get("language", "ja")
                audio = message["audio"]
                if not isinstance(language, str) or language not in LANGUAGES:
                    await send_error(websocket, "unsupported_language", "The requested language is not supported.")
                    return
                if (
                    not valid_object(audio, {"encoding", "sampleRateHz", "channels"})
                    or audio["encoding"] != "pcm_s16le"
                    or audio["sampleRateHz"] != SAMPLE_RATE
                    or audio["channels"] != 1
                ):
                    await send_error(websocket, "unsupported_audio_format", "Expected 16 kHz mono pcm_s16le audio.")
                    return
                session_ready = True
                session_language = LANGUAGES[language]
                await websocket.send_json({"type": "session.ready"})
                continue

            if message_type == "utterance.start":
                if not valid_object(message, {"type", "utteranceId"}) or utterance_id is not None:
                    await send_error(websocket, "invalid_message", "Unexpected or invalid utterance.start frame.")
                    return
                new_utterance_id = message["utteranceId"]
                if not isinstance(new_utterance_id, str) or not new_utterance_id:
                    await send_error(websocket, "invalid_message", "utteranceId must be a non-empty string.")
                    return
                utterance_id = new_utterance_id
                last_partial_text = ""
                async with inference_lock:
                    state = await asyncio.to_thread(
                        asr.init_streaming_state,
                        language=session_language,
                        chunk_size_sec=2.0,
                    )
                continue

            if message_type == "utterance.end":
                if (
                    not valid_object(message, {"type", "utteranceId"})
                    or utterance_id is None
                    or message["utteranceId"] != utterance_id
                ):
                    await send_error(websocket, "invalid_message", "Unexpected or invalid utterance.end frame.")
                    return
                completed_utterance_id = utterance_id
                async with inference_lock:
                    await asyncio.to_thread(asr.finish_streaming_transcribe, state)
                    text = state.text
                await websocket.send_json({
                    "type": "transcript.final",
                    "utteranceId": completed_utterance_id,
                    "text": text,
                })
                utterance_id = None
                state = None
                continue

            await send_error(websocket, "invalid_message", "Unexpected control frame.")
            return
    except WebSocketDisconnect:
        return
    except Exception:
        logger.exception("ASR WebSocket session failed")
        try:
            await send_error(websocket, "inference_failed", "Speech recognition failed.")
        except (RuntimeError, WebSocketDisconnect):
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
