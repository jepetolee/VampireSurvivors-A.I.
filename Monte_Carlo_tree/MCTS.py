import numpy as np
import math
import collections
import random


class Node:
    def __init__(self):
        self.leaf = True
        self.terminal = False

    def expand(self, moves):
        m = len(moves)
        if m == 0:
            self.terminal = True
        else:
            self.S = numpy.zeros(m)

            self.T = numpy.full(m, 0.001)

            self.moves = moves

            self.children = [Node() for a in range(m)]
            self.leaf = False

    def update(self, idx, score):
        self.S[idx] += score
        self.T[idx] += 1

    def choose(self):
        idx = numpy.argmax(self.S / self.T + numpy.sqrt(2.0 / self.T * numpy.log(1 + self.T.sum())))
        return idx


class MCTS_Node:
    def __init__(self):
        self.node =Node()
        self.children = list()

    def search(self,items):
        self.mcts(self.node,items)
        idx = numpy.argmax(self.node.T)
        move = node.moves[idx]
        return move

    def mcts(self, node,items):
        if node.leaf:
            node.expand(items)
            rollout = True
        else:
            rollout = False

        if node.terminal:
            return 0

        idx = node.choose()
        move = node.moves[idx]
        if self.game.make_move(move):
            val = 1
        elif rollout:
            val = -self.rollout(items)
        else:
            val = -self.mcts(node.children[idx],items)

        node.update(idx, val)
        return val

    def rollout(self,items):
        moves = items
        if len(moves) == 0:  # game is drawn
            return 0
        move = random.choice(moves)
        if self.game.make_move(move):  # current player won
            val = 1
        else:
            val = -self.rollout(items)
        self.game.unmake_move()
        return val



class MCTS:
    def __init__(self):
        self.Node = MCTS_Node()

    def input(self, items):
        self.Node.search(items)
        return 0

    def append_reward(self, reward):
#        self.Node.
        return 0
