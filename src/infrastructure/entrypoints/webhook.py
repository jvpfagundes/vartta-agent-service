"""Webhook do WhatsApp (WAHA).

Fluxo de uma mensagem recebida:
1. recebe o webhook, valida o segredo, extrai os dados e descarta duplicadas;
2. (em background) resolve o tenant, e transcreve o áudio quando necessário;
3. bufferiza o texto no `ConversationDebouncer` (em memória, por conversa);
4. espera `MESSAGE_DEBOUNCE_SECONDS` sem novas mensagens; se chegar outra durante a espera
   ou durante a execução do agente, o ciclo atual é cancelado (com rollback do checkpoint) e um
   novo ciclo agrega todos os fragmentos em uma única mensagem;
5. executa o agente com o texto agregado e, após `guard.commit()` (a partir daí o ciclo não é
   mais interrompível), responde ao cliente em várias mensagens separadas.
"""

import asyncio
import base64
from collections import OrderedDict
import hmac
import logging
import re
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from src.application.services.conversation_debouncer import ConversationDebouncer, CycleGuard
from src.application.services.reminder_choice import parse_reminder_choice
from src.domain.conversation.agent_settings import parse_agent_style, resolve_custom_instructions
from src.domain.conversation.models import TenantWhatsAppConfig, WhatsAppConversationContext
from src.infrastructure.adapters.audio.groq_transcriber import GroqAudioTranscriber
from src.infrastructure.adapters.db.whatsapp_repository import get_tenant_config_by_instance
from src.infrastructure.adapters.http.schedula_client import HttpSchedulaClient
from src.infrastructure.adapters.http.waha_client import WahaClient
from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/webhook", tags=["webhook"])

_DEDUP_CACHE_MAX = 1000
_processed_messages: OrderedDict[str, bool] = OrderedDict()

# Estado em memória (1 worker): agrupa mensagens fragmentadas e cancela o ciclo em andamento.
_debouncer = ConversationDebouncer(settings.MESSAGE_DEBOUNCE_SECONDS)


def get_debouncer() -> ConversationDebouncer:
    return _debouncer


class WebhookResponse(BaseModel):
    status: str = "received"


def _is_duplicate_message(message_id: str) -> bool:
    if not message_id:
        return False
    if message_id in _processed_messages:
        return True
    _processed_messages[message_id] = True
    while len(_processed_messages) > _DEDUP_CACHE_MAX:
        _processed_messages.popitem(last=False)
    return False


def _is_ignored_chat(chat_id: str) -> bool:
    if not chat_id:
        return True
    lower = chat_id.lower()
    return (
        lower.endswith("@g.us")
        or lower.endswith("@broadcast")
        or lower.endswith("@newsletter")
        or lower == "status@broadcast"
    )


def _extract_phone_from_chat_id(chat_id: str, participant_alt: str = "") -> str:
    source = participant_alt if ("@lid" in chat_id and participant_alt) else chat_id
    head = source.split("@")[0].split(":")[0]
    return re.sub(r"\D", "", head)


def _extract_webhook_data(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extrai dados padronizados tanto do formato nativo do WAHA ('message') quanto do formato 'messages.upsert'."""
    event = payload.get("event", "")

    # 1. Formato nativo do WAHA: {"event": "message", "session": "...", "payload": {...}}
    if event == "message":
        session_name = str(payload.get("session") or "").strip()
        msg = payload.get("payload") or {}
        if not isinstance(msg, dict) or not session_name:
            return None

        if msg.get("fromMe"):
            return None

        chat_id = str(msg.get("from") or "").strip()
        if _is_ignored_chat(chat_id):
            return None

        raw_data = msg.get("_data") if isinstance(msg.get("_data"), dict) else {}
        customer_name = str(
            raw_data.get("notifyName")
            or raw_data.get("PushName")
            or msg.get("notifyName")
            or ""
        ).strip()

        remote_jid_alt = str(
            raw_data.get("key", {}).get("remoteJidAlt")
            or raw_data.get("Info", {}).get("SenderAlt")
            or ""
        )
        customer_phone = _extract_phone_from_chat_id(chat_id, remote_jid_alt)
        message_id = str(msg.get("id") or "").strip()

        message_text = str(msg.get("body") or "").strip()
        media = msg.get("media") if isinstance(msg.get("media"), dict) else {}
        media_url = str(media.get("url") or "").strip()
        media_base64 = str(media.get("data") or "").strip()
        media_mimetype = str(media.get("mimetype") or "").strip().lower()
        msg_type = str(msg.get("type") or raw_data.get("Type") or "").strip().lower()

        is_audio = (
            media_mimetype.startswith("audio/")
            or media_mimetype == "application/ogg"
            or msg_type in ("ptt", "audio")
        ) and bool(media_url or media_base64)

        if not message_text and not is_audio:
            return None

        return {
            "session_name": session_name,
            "chat_id": chat_id,
            "customer_phone": customer_phone,
            "customer_name": customer_name,
            "message_id": message_id,
            "message_text": message_text,
            "is_audio": is_audio,
            "media_url": media_url,
            "media_base64": media_base64,
            "media_mimetype": media_mimetype or "audio/ogg",
        }

    # 2. Formato alternativo (Evolution / WAHA GOWS messages.upsert)
    if event == "messages.upsert":
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            return None

        key = data.get("key") or {}
        if key.get("fromMe"):
            return None

        chat_id = str(key.get("remoteJid") or "").strip()
        if _is_ignored_chat(chat_id):
            return None

        session_name = str(
            payload.get("session")
            or payload.get("instanceName")
            or (payload.get("instance") or {}).get("instanceId")
            or (payload.get("instance") or {}).get("instanceName")
            or ""
        ).strip()
        if not session_name:
            return None

        customer_name = str(data.get("pushName") or "").strip()
        customer_phone = _extract_phone_from_chat_id(
            chat_id, str(key.get("remoteJidAlt") or "")
        )
        message_id = str(key.get("id") or "").strip()

        message = data.get("message") or {}
        message_text = str(
            message.get("conversation")
            or (message.get("extendedTextMessage") or {}).get("text")
            or ""
        ).strip()

        audio_msg = message.get("audioMessage") or {}
        media_url = str(audio_msg.get("url") or data.get("mediaUrl") or "").strip()
        media_base64 = str(message.get("base64") or data.get("base64") or "").strip()
        media_mimetype = str(audio_msg.get("mimetype") or "audio/ogg").strip().lower()
        is_audio = bool(audio_msg) and bool(media_url or media_base64)

        if not message_text and not is_audio:
            return None

        return {
            "session_name": session_name,
            "chat_id": chat_id,
            "customer_phone": customer_phone,
            "customer_name": customer_name,
            "message_id": message_id,
            "message_text": message_text,
            "is_audio": is_audio,
            "media_url": media_url,
            "media_base64": media_base64,
            "media_mimetype": media_mimetype,
        }

    return None


async def _handle_reminder_reply(
    tenant_config: TenantWhatsAppConfig,
    chat_id: str,
    customer_phone: str,
    choice: str,
) -> bool:
    """Confirma/cancela o agendamento aguardando resposta e avisa o cliente.

    Devolve True se a mensagem foi tratada. Se o número não tem agendamento aguardando
    confirmação (ou o schedula_back falhar), devolve False e a mensagem segue o fluxo normal,
    sem alterar nenhum agendamento.
    """
    if not customer_phone:
        return False

    schedula_client = HttpSchedulaClient(
        tenant_id=tenant_config.tenant_id, user_id=customer_phone
    )
    try:
        outcome = await schedula_client.reply_to_reminder(customer_phone, choice)
    except Exception:
        logger.exception(
            "Failed to process reminder reply (tenant=%s, chat=%s)",
            tenant_config.tenant_id,
            chat_id,
        )
        return False
    finally:
        await asyncio.shield(schedula_client.close())

    reply_text = str(outcome.get("message") or "").strip()
    if outcome.get("result") == "no_pending" or not reply_text:
        return False

    logger.info(
        "Reminder reply handled (tenant=%s, chat=%s): %s",
        tenant_config.tenant_id,
        chat_id,
        outcome.get("result"),
    )
    waha_client = WahaClient()
    try:
        await waha_client.send_text(
            session_name=tenant_config.instance_name,
            chat_id=chat_id,
            text=reply_text,
        )
    except Exception:
        logger.exception(
            "Failed to send reminder reply feedback (tenant=%s, session=%s, chat=%s, phone=%s)",
            tenant_config.tenant_id,
            tenant_config.instance_name,
            chat_id,
            customer_phone,
        )
        try:
            fallback_chat_id = await waha_client.resolve_chat_id(
                tenant_config.instance_name, customer_phone
            )
            if fallback_chat_id and fallback_chat_id != chat_id:
                await waha_client.send_text(
                    session_name=tenant_config.instance_name,
                    chat_id=fallback_chat_id,
                    text=reply_text,
                    simulate_typing=False,
                )
                logger.info(
                    "Reminder reply feedback delivered via fallback chat_id (tenant=%s, fallback_chat=%s)",
                    tenant_config.tenant_id,
                    fallback_chat_id,
                )
        except Exception:
            logger.exception(
                "Fallback reminder reply feedback also failed (tenant=%s, phone=%s)",
                tenant_config.tenant_id,
                customer_phone,
            )
    return True


async def _process_whatsapp_message(extracted: dict[str, Any]) -> None:

    session_name: str = extracted["session_name"]
    chat_id: str = extracted["chat_id"]
    customer_phone: str = extracted["customer_phone"]
    customer_name: str = extracted["customer_name"]
    message_text: str = extracted["message_text"]
    is_audio: bool = extracted["is_audio"]
    media_url: str = extracted["media_url"]
    media_base64: str = extracted["media_base64"]
    media_mimetype: str = extracted["media_mimetype"]

    tenant_config = await get_tenant_config_by_instance(session_name)
    if tenant_config is None:
        logger.debug("No active tenant found for WhatsApp session=%s", session_name)
        return

    # Resposta "1"/"2" a um lembrete: é determinística e vale para todo tenant com WhatsApp
    # (os lembretes são enviados mesmo sem o add-on de IA), por isso vem antes dessas checagens.
    reminder_choice = None if is_audio else parse_reminder_choice(message_text)
    if reminder_choice and await _handle_reminder_reply(
        tenant_config, chat_id, customer_phone, reminder_choice
    ):
        return

    if not tenant_config.ai_enabled:
        logger.debug(
            "WhatsApp AI automation is disabled for tenant=%s (session=%s)",
            tenant_config.tenant_id,
            session_name,
        )
        return

    if not tenant_config.addon_active:
        logger.debug(
            "Agente de IA add-on is not active for tenant=%s (session=%s)",
            tenant_config.tenant_id,
            session_name,
        )
        return

    waha_client = WahaClient()

    conversation_id = f"wa_{tenant_config.tenant_id}_{chat_id}"
    custom_prompt = resolve_custom_instructions(
        tenant_config.ai_settings, tenant_config.ai_custom_prompt
    )
    style = parse_agent_style(tenant_config.ai_settings)

    async def run_cycle(aggregated_text: str, guard: CycleGuard) -> None:
        """Um ciclo do agente: pode ser cancelado (CancelledError) até o `guard.commit()`."""
        context = WhatsAppConversationContext(
            tenant_id=tenant_config.tenant_id,
            instance_name=tenant_config.instance_name,
            chat_id=chat_id,
            customer_phone=customer_phone,
            customer_name=customer_name,
            conversation_id=conversation_id,
            business_name=tenant_config.business_name,
            custom_prompt=custom_prompt,
            style=style,
        )

        schedula_client = HttpSchedulaClient(
            tenant_id=tenant_config.tenant_id,
            user_id=customer_phone or chat_id,
        )
        try:
            from src.infrastructure.entrypoints.app import get_whatsapp_use_case

            # Dispara sinal de 'digitando...' imediatamente enquanto a IA processa
            await waha_client.start_typing(
                session_name=tenant_config.instance_name,
                chat_id=chat_id,
            )

            use_case = get_whatsapp_use_case()
            replies = await use_case.execute(
                context=context,
                user_message=aggregated_text,
                schedula_client=schedula_client,
                cycle_guard=guard,
            )

            # A partir do envio das respostas o ciclo não pode mais ser interrompido
            # (evita resposta pela metade e mensagens duplicadas).
            guard.commit()

            parts = [part for part in replies if part]
            sent_parts = 0
            for part in parts:
                try:
                    await waha_client.send_text(
                        session_name=tenant_config.instance_name,
                        chat_id=chat_id,
                        text=part,
                    )
                except Exception:
                    logger.exception(
                        "Failed to send WhatsApp reply part %d/%d for conversation %s; "
                        "skipping remaining parts",
                        sent_parts + 1,
                        len(parts),
                        conversation_id,
                    )
                    break
                sent_parts += 1
            if sent_parts:
                logger.info(
                    "Sent %d/%d WhatsApp AI reply part(s) for conversation %s: %s",
                    sent_parts,
                    len(parts),
                    conversation_id,
                    parts[0][:120],
                )
            if sent_parts < len(parts):
                # O checkpoint já guarda a resposta completa; corrige para as partes entregues.
                await use_case.record_delivered_replies(context, parts[:sent_parts])
        except Exception:
            # CancelledError herda de BaseException e, por isso, atravessa este bloco
            # (o cancelamento do ciclo não é engolido); o finally abaixo ainda fecha o cliente.
            logger.exception(
                "Failed to process or reply to WhatsApp message for conversation %s",
                conversation_id,
            )
        finally:
            await asyncio.shield(schedula_client.close())

    # Reserva a posição da mensagem no buffer já na chegada: preserva a ordem de chegada e
    # interrompe o ciclo em andamento enquanto o áudio ainda está sendo baixado e transcrito.
    slot = get_debouncer().reserve(conversation_id, run_cycle)
    try:
        # Se for mensagem de áudio, baixa a mídia e transcreve via Groq
        if is_audio and not message_text:
            try:
                if media_base64:
                    audio_bytes = base64.b64decode(media_base64)
                    content_type = media_mimetype
                else:
                    audio_bytes, content_type = await waha_client.download_media(media_url)

                transcriber = GroqAudioTranscriber()
                message_text = await transcriber.transcribe(
                    audio_bytes=audio_bytes,
                    mime_type=content_type or media_mimetype,
                )
                logger.info(
                    "Transcribed WhatsApp audio for tenant=%s, chat=%s: %s",
                    tenant_config.tenant_id,
                    chat_id,
                    message_text[:120],
                )
            except Exception:
                logger.exception(
                    "Failed to transcribe WhatsApp audio (tenant=%s, chat=%s)",
                    tenant_config.tenant_id,
                    chat_id,
                )
                try:
                    await waha_client.send_text(
                        session_name=tenant_config.instance_name,
                        chat_id=chat_id,
                        text=(
                            "Desculpe, não consegui ouvir o seu áudio agora. "
                            "Poderia escrever por texto, por favor? 😊"
                        ),
                    )
                except Exception:
                    logger.exception("Failed to send audio fallback message via WAHA")
                return

        if not message_text.strip():
            return

        slot.fill(message_text)
    finally:
        # Idempotente: se a posição não foi preenchida (falha, mensagem vazia, cancelamento),
        # libera a reserva para a conversa seguir com os demais fragmentos.
        slot.release()


@router.post("/whatsapp", response_model=WebhookResponse)
async def whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> WebhookResponse:
    if settings.WEBHOOK_SECRET:
        provided_secret = (
            request.headers.get("x-webhook-secret")
            or request.headers.get("x-api-key")
            or ""
        )
        if not hmac.compare_digest(provided_secret, settings.WEBHOOK_SECRET):
            raise HTTPException(status_code=401, detail="Unauthorized webhook")

    try:
        payload: dict[str, Any] = await request.json()
    except Exception:
        return WebhookResponse()

    extracted = _extract_webhook_data(payload)
    if extracted is None:
        return WebhookResponse()

    if _is_duplicate_message(extracted.get("message_id", "")):
        return WebhookResponse()

    background_tasks.add_task(_process_whatsapp_message, extracted)
    return WebhookResponse()
