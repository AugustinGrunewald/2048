# 2048
A reproduction of the famous game and some bots to play

---
# About the use of the bots
To choose the bot, you just have to select the bot type in the 2048.py file.

There's a first simple random bot that just plays random moves (random_bot).
The second one aim to be an evolutionary multilayer perceptron. In the evol_bot_MLP.py you can create a population and let it evolve. If you save it, it will create a folder with the training folder, you can then load the wanted bot and let it play in the 2048.py. The selection process of the population focus on the score each bot obtains by playing the game and how much moves it plays. 