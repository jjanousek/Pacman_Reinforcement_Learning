# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util
# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.025, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)

        # stores the Q-values according to coordinate.
        self.q_values = util.Counter()
        # initialize previous pacman move, previous positions and previous score
        self.prev_move=None
        self.prev_pacman = None
        self.prev_ghost = None
        self.prev_score = 0.0

        # Count the number of games we have played
        self.episodesSoFar = 0


    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    # get Q value for certain state or 0.0 if state is new
    def lookupQValue(self, pacman, ghost, action):
        if (pacman,ghost,action) in self.q_values:
            return self.q_values[(pacman,ghost,action)]
        else:
            return 0.0

    # Function to iterate over all legal actions and return the highest state-
    # action-pair Q-value
    def maxQ(self, state, pacman, ghost):

        legal_actions = state.getLegalPacmanActions()
        # remove stopping as a an option
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)

        if not legal_actions:
            return 0.0

        maximumQ=[]
        for action in legal_actions:
            qval = self.lookupQValue(pacman,ghost,action)
            maximumQ.append(qval)
        return max(maximumQ)

    # Funtion to look for the highest Q-values accessible through legal actions
    # surrounding Pacman. Should there be multiple actions with the same maximum
    # Q-value, the function makes a random choice.
    def qValuetoAction(self, state, pacman, ghost):

        best_Q = self.maxQ(state, pacman, ghost)
        legal_actions = state.getLegalPacmanActions()

        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)

        if not legal_actions:
            return None
        best_actions=[]
        for action in legal_actions:
            if self.lookupQValue(pacman, ghost, action) == best_Q:
                best_actions.append(action)
        best_move=None
        best_move = random.choice(best_actions)

        return best_move


    def getAction(self, state):

        pacman=state.getPacmanPosition()
        # if map contains ghost, use coordinate, else stationary coordinate (0,0)
        if state.getGhostPositions()[0]:
            ghost=state.getGhostPositions()[0]
        else:
            ghost = (0,0)

        score = state.getScore()

        legal = state.getLegalPacmanActions()
        # again remove stop from legal actions
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        if not legal:
            move = None
        # The exploration part of the QLearner. When less or equal to the given
        # epsilon make a random move to explore to one of the legal nodes
        elif random.uniform(0, 1) <= self.epsilon:
            move = random.choice(legal)
        # Exploit by making a move to maximum Q-value on legal node.
        else:
            move = self.qValuetoAction(state, pacman, ghost)

        # Updates the Q-values following the update rule. The update occurs
        # once Pacman has made his move and updates the previous node with the
        # previous score as reward and the maximum Q-value from the new node
        self.q_values[(self.prev_pacman, self.prev_ghost, self.prev_move)] = \
        self.q_values[(self.prev_pacman, self.prev_ghost, self.prev_move)] + \
        self.alpha * (self.prev_score + self.gamma * self.maxQ(state, pacman, ghost) - \
        self.q_values[(self.prev_pacman, self.prev_ghost, self.prev_move)])

        # save the move, score and coordinates for the next step
        self.prev_move = move
        self.prev_pacman = pacman
        self.prev_ghost = ghost
        self.prev_score = score

        return move


    def final(self, state):

        score = state.getScore()

        # Update following the update rule. However using the final reward as
        # the maximum Q-value as there is no further step and the game ends here.
        self.q_values[(self.prev_pacman, self.prev_ghost, self.prev_move)] = \
        self.q_values[(self.prev_pacman, self.prev_ghost, self.prev_move)] + \
        self.alpha * (self.prev_score + self.gamma * score - \
        self.q_values[(self.prev_pacman, self.prev_ghost, self.prev_move)])

        # reset the internalized varibles for the next game
        self.prev_move = None
        self.prev_pacman = None
        self.prev_ghost = None
        self.prev_score = 0.0

        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)
