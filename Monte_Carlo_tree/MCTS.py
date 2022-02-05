import numpy as np
import math
import collections
import random


class Node:
    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)

    def add_virtual_loss(self, up_to=None): pass

    def revert_virtual_loss(self, up_to=None): pass

    def revert_visits(self, up_to=None): pass

    def backup_value(self, value, up_to=None): pass


class MCTS_Node:
    def __init__(self, state, n_actions, TreeEnv, action=None, parent=None):
        self.tree = TreeEnv
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

        return self.W / (1 + self.N)

    @property
    def child_Q(self):
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):
        return 1.54 * math.sqrt(1 + self.N) * self.child_prior / (1 + self.child_N)

    @property
    def child_action_score(self):
        return self.child_Q + self.child_U

    def select_leaf(self):
        current = self
        while True:
            current.N += 1
            if not current.is_expanded:
                break
            best_move = np.argmax(current.child_action_score)
            current = current.maybe_add_child(best_move)
        return current

    def may_be_child(self, action):
        if action not in self.children:
            newstate = self.tree.next_state(self.state, action)
            self.children[action] = MCTS_Node(newstate, self.n_actions,
                                         self.tree, action=action, parent=self)

        return self.children[action]

    def add_virtual_loss(self, up_to):
        self.n_vlosses += 1
        self.W -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        self.n_vlosses -= 1
        self.W += 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_virtual_loss(up_to)

    def revert_visits(self, up_to):
        self.N -= 1
        if self.parent is None or self is up_to:
            return

        self.parent.revert_visits(up_to)

    def incorporate_estimates(self, action_probs, value, up_to):

        if self.is_expanded:
            self.revert_visits(up_to=up_to)
            return
        self.is_expanded = True

        self.original_prior = self.child_prior = action_probs
        self.child_W = np.ones([self.n_actions], dtype=np.float32) * value
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value, up_to)

    def visits_as_probs(self,squash=False):
        probs = self.child_N
        if squash:
            probs = probs ** .95
        return probs/np.sum(probs)

class MCTS:
    def __init__(self):
        self.Node =MCTS_Node()

    def input(self,items):
        return 0
    def append_reward(self,reward):
        return 0



