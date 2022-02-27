import numpy as np
import random


class Node:
    def __init__(self):
        self.tensor = np.zeros((100, 30), np.int32)
        self.episode = np.zeros((100, 30), np.int32)
        self.sequence = 0

    def update(self, score):
        for episode in range(self.sequence):
            for idx in range(30):
                if self.episode[episode][idx] == 1:
                    self.tensor[episode][idx] += score

    def choose(self, items):

        self.items = items
        idx = 0
        for index in self.items:
            if self.tensor[self.sequence][index] > self.tensor[self.sequence][idx]:
                idx = index
        self.sequence += 1
        return idx


class MCTS_Node:
    def __init__(self):
        self.node = Node()
        self.idx = 0
        self.sequence = self.node.sequence

    def search(self, items, rollout=True):

        if rollout:
            rand = [0,1]
            if random.choice(rand) == 1:
                move = random.choice(items)
                self.idx = move
            else:
                self.idx = self.node.choose(items)
                move = items[self.idx]
        else:
            self.idx = self.node.choose(items)
            self.node.episode[self.sequence][self.idx] = 1
            move = items[self.idx]
        self.sequence = self.node.sequence
        return move

    def update(self, reward):
        self.node.update(reward)

    def backup(self):
        self.node.tensor = np.load("Monte_Carlo_tree/mcts.npy")

    def save(self):
        np.save("Monte_Carlo_tree/mcts", self.node.tensor)

    def mcts_vector(self):
        return self.node.tensor


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
            return False

    def backup(self):
        self.Node.backup()

    def tensor(self):
        return self.Node.mcts_vector()

    def save(self):
        self.Node.save()
