import numpy as np
import random


class Node:
    def __init__(self):
        self.tensor = np.zeros((100, 30), np.int32)
        self.episode = np.zeros((100, 30), np.int32)
        self.count = np.ones((100,30),np.int32)
        self.sequence = 0

    def update(self, score):
        for episode in range(self.sequence):
            for idx in range(30):
                # self.tensor[episode][idx] =0
                if self.episode[episode][idx] == 1:
                    self.tensor[episode][idx] += score

    def choose(self, items):
        """
        searching_algorithm = thompson sampling
        https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
        """
        self.items = items
        samples = np.random.normal(loc=self.tensor[self.sequence], scale=( 1 / np.sqrt(self.count[self.sequence]) )  )
        idx = items[0]

        for i in range(len(items)):
            if samples[items[i]]>samples[idx]:
                idx = items[i]

        return idx

    def backup(self):
        self.count = np.load("Monte_Carlo_tree/count.npy")
        self.tensor = np.load("Monte_Carlo_tree/mcts.npy")


class MCTS_Node:
    def __init__(self):
        self.node = Node()
        self.idx = 0
        self.sequence = self.node.sequence

    def search(self, items):

        self.idx = self.node.choose(items)

        move = 0
        for i in range(len(items)):
            if items[i] == self.idx:
                move = i
        self.node.episode[self.node.sequence][self.idx] = 1
        self.node.sequence += 1
        self.sequence += 1
        return move

    def update(self, reward):
        self.node.update(reward)

    def backup(self):
        self.node.backup()

    def save(self):
        self.node.count += self.node.episode
        np.save("Monte_Carlo_tree/count", self.node.count)
        np.save("Monte_Carlo_tree/mcts", self.node.tensor)

    def mcts_vector(self):
        return self.node.episode


class MCTS:

    def __init__(self):
        self.Node = MCTS_Node()

    def input(self, items):
        return self.Node.search(items)

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
