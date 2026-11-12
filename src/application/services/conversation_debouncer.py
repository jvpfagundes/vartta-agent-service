import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

# Limites do buffer por conversa (proteção contra crescimento sem teto).
DEFAULT_MAX_FRAGMENT_CHARS = 1500
DEFAULT_MAX_FRAGMENTS = 30
# Teto de espera total, em múltiplos do debounce, desde a chegada do fragmento mais antigo.
DEFAULT_MAX_WAIT_FACTOR = 4.0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 30.0


class CycleGuard:
    """Sinalizador de um ciclo do agente (uma instância nova por ciclo).

    Enquanto `committed` for False, o ciclo pode ser cancelado quando chegar uma mensagem nova.
    Depois de `commit()` o ciclo passa a ser ininterruptível: é chamado antes de efeitos
    colaterais que não podem ser repetidos ou deixados pela metade (criar agendamento,
    enviar as respostas ao cliente).
    """

    def __init__(self) -> None:
        self.committed = False

    def commit(self) -> None:
        self.committed = True


RunCycle = Callable[[str, CycleGuard], Awaitable[None]]


class _Fragment:
    """Posição de uma mensagem no buffer. `text` é None enquanto a mensagem está pendente
    (por exemplo, um áudio ainda sendo transcrito), o que preserva a ordem de chegada."""

    def __init__(self, text: str | None, run_cycle: RunCycle) -> None:
        self.text = text
        self.run_cycle = run_cycle
        self.arrived_at = time.monotonic()

    @property
    def pending(self) -> bool:
        return self.text is None


class FragmentSlot:
    """Reserva de posição no buffer, devolvida por `ConversationDebouncer.reserve`.

    Deve ser preenchida com `fill(texto)` ou descartada com `release()` (idempotente), de
    preferência em um `try/finally`, senão a conversa não avança.
    """

    def __init__(
        self,
        debouncer: "ConversationDebouncer",
        key: str,
        state: "_ConversationState",
        fragment: _Fragment,
    ) -> None:
        self._debouncer = debouncer
        self._key = key
        self._state = state
        self._fragment = fragment
        self._closed = False

    def fill(self, text: str) -> None:
        """Preenche a posição reservada e reinicia a janela de debounce."""
        if self._closed:
            return
        self._closed = True
        self._debouncer._fill(self._key, self._state, self._fragment, text)

    def release(self) -> None:
        """Descarta a reserva (ex.: falha na transcrição). Não faz nada se já preenchida."""
        if self._closed:
            return
        self._closed = True
        self._debouncer._release(self._key, self._state, self._fragment)


class _Cycle:
    """Ciclo agendado: a task asyncio e o guard (definido só quando o ciclo realmente começa)."""

    def __init__(self) -> None:
        self.task: asyncio.Task[None] | None = None
        self.guard: CycleGuard | None = None
        self.cancel_requested = False

    @property
    def interruptible(self) -> bool:
        return self.guard is None or not self.guard.committed


class _ConversationState:
    def __init__(self) -> None:
        self.fragments: list[_Fragment] = []
        self.cycles: list[_Cycle] = []


class ConversationDebouncer:
    """Agrupa mensagens fragmentadas por conversa e cancela o ciclo em andamento quando chega outra.

    Fluxo: cada `submit` (ou `reserve` + `fill`) acrescenta o fragmento ao buffer da conversa,
    cancela o ciclo anterior (se ainda não estiver "committed") e agenda um novo ciclo, que espera
    as tasks anteriores terminarem (inclusive o rollback do checkpoint), aguarda `debounce_seconds`
    e então executa `run_cycle` com todos os fragmentos pendentes unidos por quebra de linha. O
    buffer só é consumido quando o ciclo termina (com sucesso ou com erro); ciclos cancelados
    deixam os fragmentos no buffer para serem reagrupados no ciclo seguinte.

    Ordem e mídia lenta: `reserve` registra a posição da mensagem no momento da chegada (e já
    interrompe o ciclo em andamento); `fill` entrega o texto depois (ex.: transcrição de áudio) e
    `release` descarta a posição em caso de falha. Enquanto houver posições pendentes, nenhum ciclo
    executa; o `fill`/`release` agenda o ciclo que reagrupa tudo na ordem de chegada.

    Se o ciclo atual já está "committed" (CycleGuard.commit()), a mensagem nova NÃO o cancela:
    o novo ciclo espera o atual terminar e roda depois, apenas com os fragmentos restantes.

    Limites: cada fragmento é truncado em `max_fragment_chars` (mantém o início) e o buffer guarda
    no máximo `max_fragments` (descarta os mais antigos). A espera total desde o fragmento mais
    antigo é limitada a `max_wait_seconds` (padrão: 4x o debounce), para que um cliente que envia
    mensagens continuamente não adie o atendimento para sempre. Um ciclo que começa por ter atingido
    esse teto é "committed" desde o início: mensagens novas não o cancelam (senão cada uma geraria
    uma chamada descartada ao LLM e o cliente nunca receberia resposta); elas ficam no buffer e são
    atendidas pelo ciclo seguinte, que espera este terminar.

    Depois do `shutdown`, o debouncer fica encerrado: `submit`, `reserve`, `fill` e `release`
    deixam de agendar ciclos (o que evita tasks órfãs após o fechamento do checkpointer).

    Limitações (decisão de projeto): o estado fica em memória do processo, porque cancelar o ciclo
    em andamento exige cancelar uma asyncio.Task viva no mesmo processo. O serviço roda com 1 worker;
    um restart (ou o `shutdown`) perde os fragmentos que estavam na janela de debounce (até
    `debounce_seconds`). Escalar horizontalmente exigiria afinidade por chat_id no balanceador (ou
    um barramento pub/sub para propagar o cancelamento). O estado de uma conversa é descartado
    quando não há mais tasks vivas nem fragmentos, então o dicionário só cresce com o número de
    conversas ativas.
    """

    def __init__(
        self,
        debounce_seconds: float,
        *,
        max_wait_seconds: float | None = None,
        max_fragment_chars: int = DEFAULT_MAX_FRAGMENT_CHARS,
        max_fragments: int = DEFAULT_MAX_FRAGMENTS,
    ) -> None:
        self._debounce_seconds = max(0.0, float(debounce_seconds))
        if max_wait_seconds is None:
            max_wait_seconds = self._debounce_seconds * DEFAULT_MAX_WAIT_FACTOR
        self._max_wait_seconds = max(self._debounce_seconds, float(max_wait_seconds))
        self._max_fragment_chars = max(1, int(max_fragment_chars))
        self._max_fragments = max(1, int(max_fragments))
        self._states: dict[str, _ConversationState] = {}
        self._closed = False

    # ------------------------------------------------------------------ API pública

    def submit(self, key: str, text: str, run_cycle: RunCycle) -> None:
        """Registra um fragmento e agenda o ciclo. Síncrono de propósito: não há await antes de
        alterar o estado, então submits concorrentes nunca se intercalam."""
        if self._closed:
            logger.warning("Debouncer is shut down; message of conversation %s ignored", key)
            return
        state = self._get_state(key)
        state.fragments.append(_Fragment(self._limit_text(key, text), run_cycle))
        self._trim_fragments(key, state)
        self._schedule(key, state, run_cycle)

    def reserve(self, key: str, run_cycle: RunCycle) -> FragmentSlot:
        """Reserva a posição de uma mensagem cujo texto ainda não está pronto (ex.: áudio).

        Já cancela o ciclo interrompível em andamento, como um `submit`. Síncrono.
        """
        if self._closed:
            logger.warning("Debouncer is shut down; reservation of conversation %s ignored", key)
            # Reserva inerte: fill/release não fazem nada.
            slot = FragmentSlot(self, key, _ConversationState(), _Fragment(None, run_cycle))
            slot._closed = True
            return slot
        state = self._get_state(key)
        fragment = _Fragment(None, run_cycle)
        state.fragments.append(fragment)
        self._trim_fragments(key, state)
        self._schedule(key, state, run_cycle)
        return FragmentSlot(self, key, state, fragment)

    # ------------------------------------------------------------------ internos

    def _get_state(self, key: str) -> _ConversationState:
        state = self._states.get(key)
        if state is None:
            state = _ConversationState()
            self._states[key] = state
        return state

    def _limit_text(self, key: str, text: str) -> str:
        if len(text) > self._max_fragment_chars:
            logger.warning(
                "Fragment of conversation %s truncated from %d to %d chars",
                key,
                len(text),
                self._max_fragment_chars,
            )
            return text[: self._max_fragment_chars]
        return text

    def _trim_fragments(self, key: str, state: _ConversationState) -> None:
        """Descarta os fragmentos preenchidos mais antigos quando o buffer passa do limite."""
        while len(state.fragments) > self._max_fragments:
            oldest = next((f for f in state.fragments if not f.pending), None)
            if oldest is None:
                return
            state.fragments.remove(oldest)
            logger.warning(
                "Buffer of conversation %s exceeded %d fragments; dropped the oldest one",
                key,
                self._max_fragments,
            )

    def _fill(self, key: str, state: _ConversationState, fragment: _Fragment, text: str) -> None:
        fragment.text = self._limit_text(key, text)
        if fragment in state.fragments:
            self._schedule(key, state, fragment.run_cycle)

    def _release(self, key: str, state: _ConversationState, fragment: _Fragment) -> None:
        if fragment in state.fragments:
            state.fragments.remove(fragment)
        if state.fragments:
            # Os demais fragmentos ficaram sem ciclo (o da reserva retorna ao ver pendências).
            self._schedule(key, state, fragment.run_cycle)
        elif not state.cycles and self._states.get(key) is state:
            del self._states[key]

    def _schedule(self, key: str, state: _ConversationState, run_cycle: RunCycle) -> None:
        """Cancela os ciclos interrompíveis e agenda um novo ciclo que espera os anteriores."""
        if self._closed:
            return
        predecessors: list[asyncio.Task[None]] = []
        for previous in state.cycles:
            task = previous.task
            if task is None or task.done():
                continue
            predecessors.append(task)
            if previous.interruptible and not previous.cancel_requested:
                previous.cancel_requested = True
                task.cancel()

        cycle = _Cycle()
        cycle.task = asyncio.create_task(
            self._run(state, cycle, predecessors, run_cycle),
            name=f"debounce-cycle:{key}",
        )
        state.cycles.append(cycle)
        cycle.task.add_done_callback(lambda _task: self._on_cycle_done(key, state, cycle))

    def _wait_delay(self, state: _ConversationState) -> float:
        """Tempo de espera: o debounce, limitado pelo teto de espera total do fragmento mais antigo."""
        if self._debounce_seconds <= 0:
            return 0.0
        if not state.fragments:
            return self._debounce_seconds
        oldest_arrival = min(f.arrived_at for f in state.fragments)
        remaining = oldest_arrival + self._max_wait_seconds - time.monotonic()
        return max(0.0, min(self._debounce_seconds, remaining))

    async def _run(
        self,
        state: _ConversationState,
        cycle: _Cycle,
        predecessors: list[asyncio.Task[None]],
        run_cycle: RunCycle,
    ) -> None:
        # Espera as tasks anteriores (inclusive as canceladas que ainda fazem rollback)
        # para que dois ciclos da mesma conversa nunca rodem ao mesmo tempo.
        if predecessors:
            await asyncio.wait(predecessors)

        delay = self._wait_delay(state)
        # Teto de espera atingido: o delay foi encurtado abaixo do debounce.
        reached_max_wait = self._debounce_seconds > 0 and delay < self._debounce_seconds
        if delay > 0:
            await asyncio.sleep(delay)

        if any(fragment.pending for fragment in state.fragments):
            # Há mensagem ainda sendo preparada (ex.: áudio em transcrição): o `fill`/`release`
            # dela agenda o ciclo que reagrupa tudo na ordem de chegada.
            return

        snapshot = list(state.fragments)
        if not snapshot:
            return
        aggregated_text = "\n".join(fragment.text or "" for fragment in snapshot)
        guard = CycleGuard()
        cycle.guard = guard
        if reached_max_wait:
            # Garante progresso a quem envia mensagens continuamente: este ciclo não é mais
            # cancelado; as mensagens novas ficam no buffer para o ciclo seguinte.
            guard.commit()

        try:
            await run_cycle(aggregated_text, guard)
        except asyncio.CancelledError:
            # Não consome o buffer: os fragmentos serão reagrupados pelo ciclo seguinte.
            raise
        except Exception:
            logger.exception("Debounced conversation cycle failed")
            if cycle.cancel_requested:
                # Ciclo cancelado cujo cancelamento foi substituído por outra exceção (por exemplo,
                # no `finally` do run_cycle): invariante "ciclo cancelado nunca consome o buffer".
                logger.warning("Cancelled cycle failed during cleanup; keeping fragments for the next cycle")
                return
            # Consome os fragmentos para não entrar em loop reprocessando a mesma entrada.

        consumed = {id(fragment) for fragment in snapshot}
        state.fragments[:] = [f for f in state.fragments if id(f) not in consumed]

    def _on_cycle_done(self, key: str, state: _ConversationState, cycle: _Cycle) -> None:
        if cycle in state.cycles:
            state.cycles.remove(cycle)
        if not state.cycles and not state.fragments and self._states.get(key) is state:
            del self._states[key]

    async def join(self) -> None:
        """Aguarda todos os ciclos (inclusive os agendados durante a espera) terminarem."""
        while True:
            tasks = [
                cycle.task
                for state in list(self._states.values())
                for cycle in state.cycles
                if cycle.task is not None and not cycle.task.done()
            ]
            if not tasks:
                return
            await asyncio.wait(tasks)

    async def shutdown(self, timeout: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS) -> None:
        """Cancela os ciclos interrompíveis e aguarda as tasks restantes por até `timeout` segundos.

        Os fragmentos que estavam na janela de debounce (ainda não processados) são descartados:
        é uma perda conhecida no desligamento. Ciclos "committed" (envio de respostas em andamento)
        recebem até `timeout` segundos para terminar; depois disso são cancelados.
        """
        self._closed = True
        tasks: list[asyncio.Task[None]] = []
        for state in list(self._states.values()):
            for cycle in list(state.cycles):
                task = cycle.task
                if task is None or task.done():
                    continue
                tasks.append(task)
                if cycle.interruptible and not cycle.cancel_requested:
                    cycle.cancel_requested = True
                    task.cancel()

        discarded = sum(
            1 for state in self._states.values() for f in state.fragments if not f.pending
        )
        if discarded:
            logger.warning(
                "Debouncer shutdown: %d buffered fragment(s) not yet processed will be discarded",
                discarded,
            )

        if tasks:
            _, still_running = await asyncio.wait(tasks, timeout=max(0.0, timeout))
            if still_running:
                logger.warning(
                    "Debouncer shutdown timed out after %.1fs; cancelling %d running cycle(s)",
                    timeout,
                    len(still_running),
                )
                for task in still_running:
                    task.cancel()
                await asyncio.gather(*still_running, return_exceptions=True)
        self._states.clear()
