import numpy as np
import math
import collections
import random


class Node:
    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)


class MCTS:
    def __init__(self, state, n_actions, tree, action=None, parent=None):
        self.tree = tree
        if parent is None:
            self.depth = 0
            parent = Node()
        else:
            self.depth = parent.depth + 1
        self.parent = parent
        self.action = action
        self.state = state
        self.n_actions = n_actions
        self.is_expanded = False
        self.n_vlosses = 0
        self.child_N = np.zeros([n_actions], dtype=np.float32)
        self.child_W = np.zeros([n_actions], dtype=np.float32)
        self.original_prior = np.zeros([n_actions], dtype=np.float32)
        self.child_prior = np.zeros([n_actions], dtype=np.float32)
        self.children = {}
    @property
    def N(self):
        return self.parent.child_N[self.action]
    @N.setter
    def N(self, value):
        self.parent.child_N = value

    @property
    def W(self):
        return self.parent.child_W[self.action]

    @W.setter
    def W(self, value):
        self.parent.child_W = value

    @property
    def Q(self):

        return self.W / (1+self.N)
    @property
    def child_Q(self):
        return self.child_W/(1+self.child_N)