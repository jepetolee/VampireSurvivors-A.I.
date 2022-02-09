# VampireSurvivorsRL

This is the project which plays Vampire Survivors using reinforcement algorithms.

## A2C
I applyed a2c(advantaged actor critic) in training those games, it performed high effieciency for training Atari-100.
#### what's actor-critic?
http://incompleteideas.net/papers/barto-sutton-anderson-83.pdf
https://arxiv.org/abs/1611.01224
### reward
I set reward for time(sec/10) in this project, because most of rogue-like genres' is deeply related to surviving time.
### without preprocessing
I'd tried to preprocessing settings with cv.templatematch,but it takes 2-3seconds so I discarded this one, maybe  apply some latest mtm algorithms to next project.
### using convolutional network
I used cnn training this model, I think it needs more efficient networks

## MCTS
Also this game needs to select items, I applyed mcts.

