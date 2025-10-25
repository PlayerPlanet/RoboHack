from __future__ import annotations
import asyncio
from typing import Any, Dict, List


class EventRouter:
    """Simple in-process pub/sub router using asyncio.Queue.

    - subscribe(topic) -> asyncio.Queue: returns a queue that will receive all messages published to topic.
    - unsubscribe(topic, queue): remove the subscriber queue.
    - publish(topic, message): publish message to all subscriber queues (non-blocking; drops when full).

    Designed to be lightweight and scoped per service.
    """

    def __init__(self) -> None:
        self._topics: Dict[str, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, topic: str, maxsize: int = 0) -> asyncio.Queue:
        """Subscribe to a topic. Returns an asyncio.Queue instance.

        The caller is responsible for draining the queue. Use `unsubscribe` to stop receiving.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        async with self._lock:
            self._topics.setdefault(topic, []).append(q)
        return q

    async def unsubscribe(self, topic: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            lst = self._topics.get(topic)
            if not lst:
                return
            try:
                lst.remove(queue)
            except ValueError:
                return
            if not lst:
                # remove empty topic
                del self._topics[topic]

    async def publish(self, topic: str, message: Any) -> None:
        """Publish a message to all subscribers of a topic.

        This operation is non-blocking for the publisher; if a subscriber queue is full the message
        will be dropped for that subscriber to avoid publisher-side backpressure.
        """
        # snapshot subscribers under lock
        async with self._lock:
            queues = list(self._topics.get(topic, []))

        for q in queues:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                # drop the message for this subscriber
                continue

    async def publish_with_backpressure(self, topic: str, message: Any, timeout: float | None) -> None:
        """Publish a message to all subscribers, waiting up to `timeout` seconds for each subscriber's
        queue to accept the message. If `timeout` is None the call behaves like the default non-blocking
        publish (attempt put_nowait and drop on full). If `timeout` is a float, this call will await
        `queue.put()` for up to `timeout` seconds per subscriber; if that expires the message is dropped
        for that subscriber.

        Note: this provides a simple per-subscriber backpressure strategy. It waits independently for
        each subscriber up to `timeout`. A more advanced strategy could wait globally or apply retries.
        """
        async with self._lock:
            queues = list(self._topics.get(topic, []))

        for q in queues:
            if timeout is None:
                try:
                    q.put_nowait(message)
                except asyncio.QueueFull:
                    continue
            else:
                try:
                    # await put with timeout per subscriber
                    await asyncio.wait_for(q.put(message), timeout=timeout)
                except asyncio.TimeoutError:
                    # drop for this subscriber on timeout
                    continue

    async def topics(self) -> Dict[str, int]:
        """Return a snapshot mapping topic -> subscriber count."""
        async with self._lock:
            return {t: len(lst) for t, lst in self._topics.items()}

    # Convenience async iterator
    async def subscribe_iter(self, topic: str, maxsize: int = 0):
        """Async generator that yields messages for the given topic until cancelled.

        Usage:
            async for msg in router.subscribe_iter("topic"):
                ...
        """
        q = await self.subscribe(topic, maxsize=maxsize)
        try:
            while True:
                msg = await q.get()
                yield msg
        finally:
            await self.unsubscribe(topic, q)
