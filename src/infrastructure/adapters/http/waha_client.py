import asyncio
import logging
import random
import re
from urllib.parse import urlparse, urlunparse

import httpx

from src.infrastructure.adapters.db.whatsapp_repository import get_waha_credentials_from_db
from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)

_MAX_MEDIA_BYTES = 15 * 1024 * 1024  # 15 MB


class WahaClient:
    """Cliente HTTP assíncrono para comunicação com a API do WAHA."""

    def __init__(self) -> None:
        self._base_url: str | None = None
        self._api_key: str | None = None

    async def _ensure_credentials(self) -> tuple[str, str]:
        if self._base_url and self._api_key:
            return self._base_url, self._api_key

        api_url = (settings.WAHA_API_URL or "").strip()
        api_key = (settings.WAHA_API_KEY or "").strip()

        if not api_url or not api_key:
            db_creds = await get_waha_credentials_from_db()
            if db_creds:
                api_url, api_key = db_creds

        if not api_url or not api_key:
            raise RuntimeError(
                "WAHA is not configured. Set WAHA_API_URL and WAHA_API_KEY or configure 'waha' in public.system_integration_keys."
            )

        api_url = api_url.rstrip("/")
        self._base_url = api_url if api_url.endswith("/api") else f"{api_url}/api"
        self._api_key = api_key
        return self._base_url, self._api_key

    async def start_typing(self, session_name: str, chat_id: str) -> None:
        """Envia sinal de 'digitando...' para a conversa no WhatsApp."""
        try:
            base_url, api_key = await self._ensure_credentials()
            headers = {
                "X-Api-Key": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{base_url}/startTyping",
                    headers=headers,
                    json={"session": session_name, "chatId": chat_id},
                )
        except Exception:
            logger.debug("Failed to send startTyping to session=%s, chat=%s", session_name, chat_id)

    async def stop_typing(self, session_name: str, chat_id: str) -> None:
        """Interrompe o sinal de 'digitando...' na conversa."""
        try:
            base_url, api_key = await self._ensure_credentials()
            headers = {
                "X-Api-Key": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{base_url}/stopTyping",
                    headers=headers,
                    json={"session": session_name, "chatId": chat_id},
                )
        except Exception:
            logger.debug("Failed to send stopTyping to session=%s, chat=%s", session_name, chat_id)

    async def send_text(
        self,
        session_name: str,
        chat_id: str,
        text: str,
        simulate_typing: bool = True,
    ) -> dict:
        """Envia mensagem de texto pelo WAHA para o chatId informado com simulação humanizada de digitação."""
        base_url, api_key = await self._ensure_credentials()
        headers = {
            "X-Api-Key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if simulate_typing and text:
            await self.start_typing(session_name, chat_id)

            # Cálculo humanizado de delay anti-bloqueio:
            # ~25ms por caractere + jitter aleatório (0.5s a 1.2s)
            # Limites: mínimo 1.5s, máximo 6.0s (evita estourar timeouts)
            char_delay = len(text) * 0.025
            jitter = random.uniform(0.5, 1.2)
            delay = max(1.5, min(6.0, char_delay + jitter))
            await asyncio.sleep(delay)

            await self.stop_typing(session_name, chat_id)

        async with httpx.AsyncClient(timeout=30.0) as client:
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    response = await client.post(
                        f"{base_url}/sendText",
                        headers=headers,
                        json={
                            "session": session_name,
                            "chatId": chat_id,
                            "text": text,
                        },
                    )
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    status = exc.response.status_code
                    if status < 500 and status != 429:
                        raise
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_exc = exc

                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))

            assert last_exc is not None
            raise last_exc

    async def resolve_chat_id(self, session_name: str, phone_number: str) -> str | None:
        """Resolve o chatId correto no WAHA a partir do telefone do cliente (fallback para @lid/JID inválido)."""
        digits = re.sub(r"\D", "", str(phone_number or ""))
        if not digits:
            return None

        if len(digits) in (10, 11) and not re.fullmatch(r"1[2-9]\d{2}[2-9]\d{6}", digits):
            digits = f"55{digits}"

        if digits.startswith("55"):
            try:
                base_url, api_key = await self._ensure_credentials()
                headers = {
                    "X-Api-Key": api_key,
                    "Accept": "application/json",
                }
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        f"{base_url}/contacts/check-exists",
                        headers=headers,
                        params={"phone": digits, "session": session_name},
                    )
                    if 200 <= response.status_code < 300:
                        data = response.json()
                        if isinstance(data, dict) and data.get("numberExists") and data.get("chatId"):
                            return str(data["chatId"])
            except Exception:
                logger.debug(
                    "Failed to resolve chatId via WAHA check-exists (session=%s, phone=%s)",
                    session_name,
                    digits,
                )

        return f"{digits}@c.us"

    async def download_media(self, media_url: str) -> tuple[bytes, str]:
        """Baixa arquivo de mídia (ex: áudio) do WAHA aplicando autenticação e limite de tamanho."""
        base_url, api_key = await self._ensure_credentials()

        # Se o WAHA enviou uma URL com localhost/127.0.0.1 ou caminho relativo,
        # reescreve para usar o host configurado no WAHA_API_URL.
        waha_root = base_url[:-4] if base_url.endswith("/api") else base_url
        parsed_waha = urlparse(waha_root)
        parsed_media = urlparse(media_url)

        if not parsed_media.scheme:
            resolved_url = f"{waha_root}/{media_url.lstrip('/')}"
        elif parsed_media.hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
            resolved_url = urlunparse(
                (
                    parsed_waha.scheme,
                    parsed_waha.netloc,
                    parsed_media.path,
                    parsed_media.params,
                    parsed_media.query,
                    parsed_media.fragment,
                )
            )
        else:
            resolved_url = media_url

        headers = {"X-Api-Key": api_key}
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(resolved_url, headers=headers)
            response.raise_for_status()
            content = response.content
            if len(content) > _MAX_MEDIA_BYTES:
                raise ValueError("Arquivo de áudio excede o tamanho máximo permitido (15 MB).")
            content_type = response.headers.get("content-type", "audio/ogg")
            return content, content_type
