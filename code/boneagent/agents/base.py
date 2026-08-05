from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar

from boneagent.domain import AgentKind, Message

Input = TypeVar("Input")
Output = TypeVar("Output")


@dataclass(frozen=True)
class AgentResponse(Generic[Output]):
    value: Output
    source_ids: tuple[str, ...]
    elapsed_seconds: float


class MessageBus:
    def __init__(self, maximum_messages: int = 100000) -> None:
        self._queues = {kind: deque[Message]() for kind in AgentKind}
        self._history: deque[Message] = deque(maxlen=maximum_messages)
        self._lock = Lock()

    def publish(self, message: Message) -> None:
        with self._lock:
            self._queues[message.receiver].append(message)
            self._history.append(message)

    def send(
        self,
        sender: AgentKind,
        receiver: AgentKind,
        message_type: str,
        cycle: int,
        payload: object,
        source_ids: tuple[str, ...] = (),
    ) -> Message:
        message = Message(
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            cycle=cycle,
            payload=payload,
            source_ids=source_ids,
            timestamp=time.time(),
        )
        self.publish(message)
        return message

    def receive(self, receiver: AgentKind, limit: int | None = None) -> list[Message]:
        with self._lock:
            queue = self._queues[receiver]
            count = len(queue) if limit is None else min(limit, len(queue))
            return [queue.popleft() for _ in range(count)]

    def history(self, cycle: int | None = None) -> tuple[Message, ...]:
        with self._lock:
            if cycle is None:
                return tuple(self._history)
            return tuple(message for message in self._history if message.cycle == cycle)

    def pending(self, receiver: AgentKind) -> int:
        with self._lock:
            return len(self._queues[receiver])


class Agent(ABC, Generic[Input, Output]):
    kind: AgentKind

    def __init__(self, bus: MessageBus) -> None:
        self.bus = bus

    @abstractmethod
    def act(self, request: Input, cycle: int) -> AgentResponse[Output]:
        raise NotImplementedError

    def notify(
        self,
        receiver: AgentKind,
        message_type: str,
        cycle: int,
        payload: object,
        source_ids: tuple[str, ...] = (),
    ) -> Message:
        return self.bus.send(
            self.kind,
            receiver,
            message_type,
            cycle,
            payload,
            source_ids,
        )
