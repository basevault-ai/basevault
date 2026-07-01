from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kernel.abstractions import LlmCall, Scheduler


class CancellationManager:
    def __init__(self):
        self._in_flight_calls: dict[str, Scheduler] = {}
        self._http_streams: dict[str, Any] = {}
        self._skipped_calls: set[str] = set()
        self._aborted: bool = False
        self._lock: Lock = Lock()

    def register_call(self, call: LlmCall, scheduler: Scheduler) -> bool:
        """Returns whether call should be executed."""
        with self._lock:
            if self._aborted or call.id in self._skipped_calls:
                return False
            self._in_flight_calls[call.id] = scheduler
            return True

    def deregister_call(self, call: LlmCall) -> None:
        with self._lock:
            if call.id in self._in_flight_calls:
                del self._in_flight_calls[call.id]

    def register_http_stream(self, call: LlmCall, stream) -> None:
        with self._lock:
            # If request is already skipped / aborted, close the stream immediately.
            if self._aborted or call.id in self._skipped_calls:
                stream.response.close()
            self._http_streams[call.id] = stream

    def deregister_http_stream(self, call: LlmCall, stream) -> None:
        with self._lock:
            if call.id in self._http_streams:
                del self._http_streams[call.id]

    def skip_call(self, id: str) -> None:
        scheduler: Scheduler | None = None
        stream: Any | None = None
        with self._lock:
            self._skipped_calls.add(id)
            if id in self._in_flight_calls:
                scheduler = self._in_flight_calls[id]
                del self._in_flight_calls[id]
            if id in self._http_streams:
                stream = self._http_streams[id]
                del self._http_streams[id]
        if scheduler:
            scheduler.abort({id}, skip=True)
        if stream:
            stream.response.close()

    def is_skipped(self, call: LlmCall) -> bool:
        with self._lock:
            return call.id in self._skipped_calls

    def abort(self) -> None:
        """Aborts all in-flight calls in this env."""
        calls: set[str] = set()
        schedulers: set[Scheduler] = set()
        streams: set[Any] = set()

        with self._lock:
            if self._aborted:
                return
            self._aborted = True
            calls = set(self._in_flight_calls.keys())
            schedulers = set(self._in_flight_calls.values())
            streams = set(self._http_streams.values())

        for scheduler in schedulers:
            scheduler.abort(calls)
        for stream in streams:
            stream.response.close()

    def aborted(self) -> bool:
        return self._aborted
