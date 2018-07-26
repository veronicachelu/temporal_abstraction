import math
import numpy as np

class PolicyIteration:
    V = None
    pi = None
    gamma = 0.9
    numStates = 0
    actionSet = None
    environment = None

    def __init__(self, gamma, env, augmentActionSet=False):
        self.gamma = gamma
        self.environment = env
        self.nb_states = env.nb_states + 1

        self.V = np.zeros(self.nb_states + 1)
        self.pi = np.zeros(self.nb_states + 1, dtype=np.int)

        if augmentActionSet:
            self.actionSet = np.append(env.get_action_set(), [4])
        else:
            self.actionSet = env.get_action_set()

    def evalPolicy(self):
        """Policy evaluation"""
        delta = 0.0
        for s in range(self.nb_states):
            v = self.V[s]
            nextS, nextR = self.environment.get_next_state_and_reward(
                s, self.pi[s])

            self.V[s] = nextR + self.gamma * self.V[nextS]
            delta = max(delta, math.fabs(v - self.V[s]))

        return delta

    def improvePolicy(self):
        """Policy improvement"""
        policy_stable = True
        for s in range(self.nb_states):
            old_action = self.pi[s]
            tempV = [0.0] * len(self.actionSet)
            # I first get all value-function estimates
            for i in range(len(self.actionSet)):
                nextS, nextR = self.environment.get_next_state_and_reward(
                    s, i)
                tempV[i] = nextR + self.gamma * self.V[nextS]

            # Now I take the argmax
            self.pi[s] = np.argmax(tempV)
            # I break ties always choosing to terminate:
            if math.fabs(tempV[self.pi[s]] - tempV[(len(self.actionSet) - 1)]) < 0.001:
                self.pi[s] = (len(self.actionSet) - 1)
            if old_action != self.pi[s]:
                policy_stable = False

        return policy_stable

    def solvePolicyIteration(self, theta=0.001):
        """Policy Iteration"""

        # Initialization is done in the constructor
        policy_stable = False

        while not policy_stable:
            # Policy evaluation
            delta = self.evalPolicy()
            while (theta < delta):
                delta = self.evalPolicy()

            # Policy improvement
            policy_stable = self.improvePolicy()

        return self.V, self.pi