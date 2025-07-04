from bot import evol_bot_MLP, random_bot

from pygame.locals import K_RIGHT, K_LEFT, K_UP, K_DOWN

from torch import nn

bots = {
    "random" : random_bot.RandomBot,
    "evolutionary_MLP" : evol_bot_MLP.EvolutionaryBotMLP
}

activations = {
    "ReLU" : nn.ReLU,
    "Tanh" : nn.Tanh,
    "Sigmoid" : nn.Sigmoid,
    None : nn.Identity
}

movements = {
    0 : K_RIGHT,
    1 : K_LEFT,
    2 : K_UP,
    3 : K_DOWN
}