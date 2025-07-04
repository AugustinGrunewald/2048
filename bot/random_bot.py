from bot import base_bot

from pygame.locals import K_RIGHT, K_LEFT, K_UP, K_DOWN

import numpy as np
from numpy.typing import NDArray

import random as rd

class RandomBot(base_bot.BaseBot):
    def __init__(self, bot_delay : int) -> None:
        super().__init__(bot_delay)

    def play_move(self, playground : NDArray[np.int64]) -> int:
        return rd.choice([K_RIGHT, K_LEFT, K_UP, K_DOWN])