import logging
import httpx

from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)

_GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

_MIMETYPE_EXTENSIONS = {
    "audio/ogg": "ogg",
    "application/ogg": "ogg",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/wav": "wav",
    "audio/webm": "webm",
}


class GroqAudioTranscriber:
    """Transcreve áudios do WhatsApp usando a API Whisper da Groq."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = (api_key or settings.GROQ_API_KEY or "").strip()
        self._model = (model or settings.GROQ_WHISPER_MODEL or "whisper-large-v3-turbo").strip()

    async def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        if not self._api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        if not audio_bytes:
            return ""

        base_mime = mime_type.split(";")[0].strip().lower()
        ext = _MIMETYPE_EXTENSIONS.get(base_mime, "ogg")
        filename = f"whatsapp_audio.{ext}"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        files = {
            "file": (filename, audio_bytes, base_mime or "audio/ogg"),
        }
        data = {
            "model": self._model,
            "language": "pt",
            "response_format": "json",
            "temperature": "0.0",
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                _GROQ_TRANSCRIPTIONS_URL,
                headers=headers,
                files=files,
                data=data,
            )
            response.raise_for_status()
            payload = response.json()
            text = payload.get("text", "")
            return str(text).strip()
