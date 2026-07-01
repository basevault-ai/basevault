import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore, Condition, Thread
from typing import override

from kernel.abstractions import LlmCall, LlmResponse, ModelSpec, Scheduler
from kernel.enums import Environment, LlmStatus
from kernel.execution_env import BoundExecutionEnv


class ThrottledScheduler(Scheduler):
    def __init__(self, model_spec: ModelSpec):
        super().__init__(model_spec)

        # TODO: Adaptive sqrt-based pool size
        pool_size: int = model_spec.max_parallelism(Environment.get())
        # Pool of workers + semaphore to wait until a slot is available.
        self.pool = ThreadPoolExecutor(max_workers=pool_size)
        self.pool_semaphore = BoundedSemaphore(pool_size)

        # Queue of work + semaphore to wait until work is available.
        self.q: deque[tuple[str, Future[LlmResponse], Callable]] = deque()
        self.retry_q: deque[tuple[str, Future[LlmResponse], Callable]] = deque()
        self.condition = Condition()

        # Start Scheduler main loop.
        self.seconds_between_requests: float = model_spec.seconds_between_requests(
            Environment.get()
        )
        self.scheduler_thread = Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

    @override
    def run(
        self, call: LlmCall, execution_env: BoundExecutionEnv, is_retry: bool
    ) -> Future[LlmResponse]:
        # Enqueue the given call and wait until it's done.
        # If <is_retry> is true, place in a high-priority retry queue.
        future: Future[LlmResponse] = Future()

        def do_work() -> None:
            try:
                for hook in execution_env.llm_hooks:
                    hook.hook_llm_started(call, execution_env)
                result: LlmResponse = self.model_spec.inference_provider().run(
                    call, execution_env
                )
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                self.pool_semaphore.release()

        with self.condition:
            if not execution_env.cancellation_manager.register_call(call, self):
                # Call is already aborted; return immediately.
                skip = execution_env.cancellation_manager.is_skipped(call)
                status = LlmStatus.SKIPPED if skip else LlmStatus.ABORTED
                future.set_result(LlmResponse.from_status(status, 0))
                return future

            if is_retry:
                self.retry_q.append((call.id, future, do_work))
            else:
                self.q.append((call.id, future, do_work))
            self.condition.notify_all()

        for hook in execution_env.llm_hooks:
            hook.hook_llm_queued(call, execution_env)
        return future

    @override
    def abort(self, calls: set[str], skip: bool = False):
        status = LlmStatus.SKIPPED if skip else LlmStatus.ABORTED
        """
        Aborts all queued calls. Already scheduled calls will be aborted by the inference provider.
        """
        with self.condition:
            new_q = deque()
            for call, future, work in self.retry_q:
                if call in calls:
                    future.set_result(LlmResponse.from_status(status, 0))
                else:
                    new_q.append((call, future, work))
            self.retry_q = new_q

            new_q = deque()
            for call, future, work in self.q:
                if call in calls:
                    future.set_result(LlmResponse.from_status(status, 0))
                else:
                    new_q.append((call, future, work))
            self.q = new_q

    def _scheduler_loop(self) -> None:
        last_scheduled_time: float = 0
        while True:
            current_time = time.monotonic()
            target_time = last_scheduled_time + self.seconds_between_requests
            if current_time < target_time:
                time.sleep(target_time - current_time)

            # Wait until pool has available
            self.pool_semaphore.acquire()

            # Wait until work is available. Prioritize the retry queue.
            with self.condition:
                while len(self.retry_q) == 0 and len(self.q) == 0:
                    self.condition.wait()
                if len(self.retry_q) > 0:
                    (call, _, work) = self.retry_q.popleft()
                else:
                    (call, _, work) = self.q.popleft()

                self.pool.submit(work)
                last_scheduled_time = time.monotonic()
