import asyncio
import json
import logging
import queue
from typing import Callable

logger = logging.getLogger(__name__)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def progress_events(work: Callable[[Callable[[str], None]], dict]):
    """Run a blocking ``work(progress)`` callable in a worker thread, streaming
    its progress messages as Server-Sent Events followed by a final ``result``
    (or ``error``) event.

    ``work`` is given a ``progress(message)`` callback and must return a
    JSON-serializable dict to be delivered as the result.
    """
    progress_queue: queue.SimpleQueue = queue.SimpleQueue()

    def progress(message: str):
        progress_queue.put({"type": "progress", "message": message})

    def drain():
        events = []
        while not progress_queue.empty():
            try:
                events.append(_sse(progress_queue.get_nowait()))
            except queue.Empty:
                break
        return events

    task = asyncio.create_task(asyncio.to_thread(work, progress))

    while not task.done():
        for event in drain():
            yield event
        await asyncio.sleep(0.1)

    for event in drain():
        yield event

    try:
        result = await task
        yield _sse({"type": "result", "data": result})
    except ConnectionError as e:
        logger.error(f"Connection error during streaming task: {e}")
        yield _sse({"type": "error", "message": str(e)})
    except Exception:
        logger.exception("Unexpected error during streaming task")
        yield _sse({"type": "error", "message": "An unexpected error occurred"})
