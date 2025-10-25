import asyncio
from common.event_router import EventRouter


def test_publish_subscribe_basic():
    async def coro():
        r = EventRouter()
        q1 = await r.subscribe("topic")
        q2 = await r.subscribe("topic")

        await r.publish("topic", "m1")
        v1 = await q1.get()
        v2 = await q2.get()
        assert v1 == "m1"
        assert v2 == "m1"

        # unsubscribe q2 and publish again
        await r.unsubscribe("topic", q2)
        await r.publish("topic", "m2")
        v1 = await q1.get()
        assert v1 == "m2"

    asyncio.get_event_loop().run_until_complete(coro())


def test_subscribe_iter_and_topics_snapshot():
    async def coro():
        r = EventRouter()
        # start a consumer that uses subscribe_iter
        async def consumer():
            got = []
            async for msg in r.subscribe_iter("t2"):
                got.append(msg)
                if len(got) >= 2:
                    break
            return got

        task = asyncio.create_task(consumer())

        # give consumer time to subscribe
        await asyncio.sleep(0)
        await r.publish("t2", 1)
        await r.publish("t2", 2)
        got = await task
        assert got == [1, 2]

        snap = await r.topics()
        # subscribe_iter unsubscribed itself after finishing -> topic may be removed
        assert isinstance(snap, dict)

    asyncio.get_event_loop().run_until_complete(coro())


def test_publish_with_backpressure_timeout():
    async def coro():
        r = EventRouter()
        # create bounded queue
        q = await r.subscribe("bp", maxsize=1)
        # fill it so a normal put would block
        await q.put("existing")

        # schedule a consumer to free a slot after 0.2s
        async def delayed_consume():
            await asyncio.sleep(0.2)
            await q.get()

        asyncio.create_task(delayed_consume())

        # publish_with_backpressure with timeout longer than consumer delay should succeed
        await r.publish_with_backpressure("bp", "msg", timeout=1.0)
        v = await q.get()
        assert v == "msg"

    asyncio.get_event_loop().run_until_complete(coro())


def test_publish_with_backpressure_timeout_expires():
    async def coro():
        r = EventRouter()
        q = await r.subscribe("bp2", maxsize=1)
        await q.put("existing")

        # publish with a very small timeout; consumer won't free the slot in time
        await r.publish_with_backpressure("bp2", "msg", timeout=0.01)
        # queue should still contain the original item and not the new one
        assert q.qsize() == 1
        v = await q.get()
        assert v == "existing"

    asyncio.get_event_loop().run_until_complete(coro())
