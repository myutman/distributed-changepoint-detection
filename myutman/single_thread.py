import abc
from typing import Any


class StreamingAlgo:

    def __init__(self, p, random_state):
        self.p = p
        self.random_state = random_state

    @abc.abstractmethod
    def process_element(self, element: float, meta: Any = None) -> None:
        pass

    @abc.abstractmethod
    def get_stat(self) -> Any:
        pass

    @abc.abstractmethod
    def test(self) -> bool:
        pass

    def restart(self) -> None:
        pass
