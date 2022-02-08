import numpy as np
import math
import collections
import random


class Node:
    def __init__(self):
        self.tensor = np.zeros((100, 20), np.int32)
        self.sequence = 0

    def expand(self, items):
        self.S = self.tensor[self.sequence]
        self.items = items

    def update(self, idx, score):
        self.S[idx] += score
        self.tensor[self.sequence] = self.S
        self.sequence += 1

    def choose(self):
        idx = 0
        for index in self.items:
            if self.S[index] > self.S[idx]:
                idx = index

        return idx


class MCTS_Node:
    def __init__(self):
        self.node = Node()
        self.idx = 0

    def search(self, items, rollout=True):

        self.node.expand(items)

        if rollout:
            move = random.choice(items)
            self.idx = move
        else:
            self.idx = self.node.choose()
            move = items[self.idx]
        return move

    def update(self, reward):
        self.node.update(self.idx, reward)

    def backup(self):
        self.node.tensor = np.load("./mcts.npy")

    def save(self):
        np.save("./mcts.npy",self.node.tensor)


class MCTS:
    def __init__(self):
        self.Node = MCTS_Node()

    def call_saves(self):
        self.Node.backup()

    def input(self, items, rollout=True):
        return self.Node.search(items, rollout)

    def append_reward(self, reward):
        self.Node.update(reward)
        self.Node.save()

    def backup(self):
        self.Node.backup()
