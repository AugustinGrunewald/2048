from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

class BaseBot(ABC):
    def __init__(self, bot_delay : int) -> None:
        super().__init__()
        self.last_move_time = 0
        self.__bot_delay = bot_delay

    @property
    def get_bot_delay(self) -> int:
        return self.__bot_delay

    @abstractmethod
    def play_move(self, playground : NDArray[np.int64]) -> int:
        pass