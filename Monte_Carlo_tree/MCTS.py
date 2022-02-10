import numpy as np
import math
import collections
import random


class Node:
    def __init__(self):
        self.tensor = np.zeros((100, 30), np.int32)
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
        self.sequence = self.node.sequence

    def search(self, items, rollout=True):

        self.node.expand(items)

        if rollout:
            rand = [1, 2, 3]
            rand = random.choice(rand)
            if rand ==1:
                move = random.choice(items)
                self.idx = move
            else:
                self.idx = self.node.choose()
                move = items[self.idx]
        else:
            self.idx = self.node.choose()
            move = items[self.idx]
        return move

    def update(self, reward):
        self.node.update(self.idx, reward)

    def backup(self):
        self.node.tensor = np.load("Monte_Carlo_tree/mcts.npy")

    def save(self):
        np.save("Monte_Carlo_tree/mcts", self.node.tensor)


class MCTS:
    def __init__(self):
        self.Node = MCTS_Node()

    def call_saves(self):
        self.Node.backup()

    def input(self, items, rollout=True):
        select = self.Node.search(items, rollout)
        for i in range(len(items)):

            if select == items[i]:
                return i

    def append_reward(self, reward):
        self.Node.update(reward)



    def checkwork(self):
        if self.Node.sequence != 0:
            return True
        else:
            False

    def backup(self):
        self.Node.backup()

    def save(self):
        self.Node.save()