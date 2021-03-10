import numpy
import random
import copt
import abc
import heuristics
import utilities
import threading

# Base Reinforcement Learning Agent that forces all RL Agents to implement the methods learn and getSolution
class BaseRLAgent:
    # Declares this class as an abstract class
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def learn(self, numberOfProblems, squareSize):
        # This method will allow the RL agent to learn
        return

    @abc.abstractmethod
    def getSolution(self, problem):
        # This method will return the computed solution to the problem
        return

# Q-Learning agent
class QLearningAgent(BaseRLAgent):

    # Constructor that initialises all the arrays and constants
    def __init__(self, numberOfPoints, numberToRound):
        # Initialises the Q-Learning dictionary/action space
        self.problems = {}
        self.actions = []
        self.success = []
        self.rewards1 = []
        self.rewards2 = []

        # Initialises the number of points that need to be connected
        self.points = numberOfPoints

        # Initialises alpha and gamma which are used in the Q-Learning algorithm
        # Alpha is the learning rate and Gamma represents how much the agent values future rewards
        self.alpha = 0.1
        self.gamma = 0.1

        # Initialises epsilon, the chance a random action is chosen in action for the agent to learn
        self.epsilon = 0.1

        # Initialises the number to round to
        self.roundNumber = numberToRound

        # Initialises the reseen problem counter
        self.reseenProblems = 0

    def __learnProblem(self, learnProblem):
        # Gets the permuted problem and the permutation
        permutedProblem, _ = self.__permuteProblem(learnProblem)

        # Stores the current points ordering
        currentOrder = []

        for point in range(0, self.points):
            # Need to get the next action a maximum of self.points times
            # If an action is unsuccessful then can break from this loop

            # Gets the next action
            nextAction = self.__getNextLearnAction(permutedProblem, currentOrder)

            # Updates the ordering
            newOrder = currentOrder + [nextAction]

            # Check to see how well the random action has done
            result = copt.evaluate(permutedProblem, newOrder)

            # Get the reward and success
            # Does 100000 - reward so that we can use the conventional Q-Learning algorithm
            # It allows for us to look at maximums rather than minimums
            reward = 100000 - result["measure"]
            success = result["success"]

            # If the action was unsuccessful, change the reward to reflect this
            if (success == 0):
                reward = 0

            # Update the arrays
            self.__updateArrays(permutedProblem, currentOrder, nextAction, success, reward)

            # Update the current order
            currentOrder = newOrder

            if (success == 0):
                # If the action was unsuccessful then can finish with this problem
                break

    # Allows the agent to learn using the specified number of problems
    def learn(self, numberOfProblems, squareSize):
        print("Learning Started")

        for i in range(0, numberOfProblems):
            validProblem = False

            # Makes sure the problem is valid
            while (validProblem is False):
                # Get the problem
                problem = copt.getProblem(self.points) if squareSize == 0 else utilities.generateSmallerProblem(self.points, squareSize)

                # Checks whether the problem is valid
                validProblem = utilities.checkValidProblem(problem) if squareSize == 0 else True

            # Create and start the threads
            thread = threading.Thread(target=self.__learnProblem, args=(problem,))
            thread.start()

            outputThread = threading.Thread(target=utilities.outputPercentageComplete, args=(i + 1, numberOfProblems, self.reseenProblems,))
            outputThread.start()

        while (thread.is_alive() or outputThread.is_alive()):
            # Waits for all threads to be completed
            pass

        print("Learning Finished")

    def __getNextLearnAction(self, problem, order):
        # Gets the next action when the agent is learning
        # Sets up the problem data to be how all problem data is stored
        problemData = str(problem + [order])

        # Checks to see if the problem has already been seen
        try:
            # Gets the index of the problem
            # This will be the index in all the other lists
            problemIndex = self.problems[problemData]

            # Sets up the arrays to store the optimal next action and optimal reward
            optimalNextAction = -1
            optimalReward = -1

            # Iterates through all the tried actions
            for i in range(0, len(self.actions[problemIndex])):
                # Want to find the action that is a combination of the one that gives the best reward
                # and the one that will allow the agent to learn a lot
                # Once the agent has learnt for a while, expectedInformation ~= 0
                # So the best action is always chosen
                expectedReward = (self.rewards1[problemIndex][i] + self.rewards2[problemIndex][i]) / 2

                # For now, if the state hasn't been seen yet then it will look at it
                expectedInformation = 100000 if ((self.success[problemIndex][i] == 1) and (expectedReward == 0)) else 0

                expectedReward = expectedReward + expectedInformation

                # Checks to see if the action was successful and checks that a higher reward was returned
                if ((self.success[problemIndex][i] == 1) and (expectedReward > optimalReward)):
                    optimalNextAction = self.actions[problemIndex][i]
                    optimalReward = expectedReward

            # Checks if a valid ordering was found
            if (optimalNextAction == -1):
                # Finds the action which will connect the closest two points not already connected
                nextAction = heuristics.ManhattanHeuristic().getNextAction(problem, order)

                # Checks to see if all the actions have been seen
                # If so, all are unsuccessful so can return any
                if (self.points == (len(permutedOrder) + len(self.actions[problemIndex]))):
                    return nextAction

                while (nextAction in self.actions[problemIndex]):
                    # Keep getting a new action until its not been seen before
                    nextAction = heuristics.RandomHeuristic().getNextAction(problem, order)

                # Found a different action to use
                return nextAction
            else:
                # Valid action found so return it
                return optimalNextAction
        except KeyError:
            # Problem not seen already so find the shortest distance between points
            return heuristics.ManhattanHeuristic().getNextAction(problem, order)

    # Gets a solution to the problem
    def getSolution(self, problem):
        # Gets the permuted problem and the permutation
        permutedProblem, permutation = self.__permuteProblem(problem)

        # Stores the current points ordering
        currentOrder = []

        # Default values for success and reward
        success = 0
        reward = 0

        for point in range(0, self.points):
            # Need to get the next action a maximum of self.points times
            # If an action is unsuccessful then can break from this loop

            # Get the best next action
            nextAction = self.__getBestNextAction(permutedProblem, currentOrder)

            # Updates the ordering
            newOrder = currentOrder + [nextAction]

            # Check to see how well the random action has done
            result = copt.evaluate(permutedProblem, newOrder)

            # Get the reward and success
            # Does 100000 - reward so that we can use the conventional Q-Learning algorithm
            # It allows for us to look at maximums rather than minimums
            reward = 100000 - result["measure"]
            success = result["success"]

            # If the action was unsuccessful, change the reward to reflect this
            if (success == 0):
                reward = 0

            # Update the arrays
            self.__updateArrays(permutedProblem, currentOrder, nextAction, success, reward)

            # Update the current order
            currentOrder = newOrder

            if (success == 0):
                # If the action was unsuccessful then can finish with this problem
                break

        # Unpermutes the order, storing -1 where there is nothing
        returnOrder = [-1]*self.points
        for i in range(0, len(currentOrder)):
            returnOrder[i] = permutation[currentOrder[i]]

        # Return the ordering, reward and success
        return returnOrder, success, reward

    # Gets the next action that the agent thinks is best
    def __getBestNextAction(self, permutedProblem, permutedOrder):
        # Sets up the problem data to be how all problem data is stored
        problemData = str(permutedProblem + [permutedOrder])

        # Checks to see if the problem has already been seen
        try:
            # Gets the index of the problem
            # This will be the index in all the other lists
            problemIndex = self.problems[problemData]

            # Sets up the arrays to store the optimal next action and optimal reward
            optimalNextAction = -1
            optimalReward = -1

            # Iterates through all the tried actions
            for i in range(0, len(self.actions[problemIndex])):
                # Checks to see if the action was successful
                # Also checks that a higher reward was returned
                expectedReward = (self.rewards1[problemIndex][i] + self.rewards2[problemIndex][i]) / 2
                if ((self.success[problemIndex][i] == 1) and (expectedReward > optimalReward)):
                    optimalNextAction = self.actions[problemIndex][i]
                    optimalReward = expectedReward

            # Checks if a valid ordering was found
            if (optimalNextAction == -1):
                # Finds the action which will connect the closest two points not already connected
                nextAction = heuristics.ManhattanHeuristic().getNextAction(permutedProblem, permutedOrder)

                # Checks to see if all the actions have been seen
                # If so, all are unsuccessful so can return any
                if (self.points == (len(permutedOrder) + len(self.actions[problemIndex]))):
                    return nextAction

                while (nextAction in self.actions[problemIndex]):
                    # Keep getting a new action until its not been seen before
                    nextAction = heuristics.RandomHeuristic().getNextAction(permutedProblem, permutedOrder)

                # Found a different action to use
                return nextAction
            else:
                # Valid action found so return it
                return optimalNextAction
        except KeyError:
            # Problem not seen already so find the shortest distance between points
            return heuristics.ManhattanHeuristic().getNextAction(permutedProblem, permutedOrder)

    # Permutes the problem so that the point with the smallest x index will be first etc
    # Returns the permuted problem as well as the permutation
    def __permuteProblem(self, problem):
        # Rounds all the points to the nearest roundNumber
        roundedProblem = [(-1, -1, -1, -1)]*self.points
        for j in range(0, self.points):
            roundedProblem[j] = (self.roundNumber * round(problem[j][0] / float(self.roundNumber)),
                                 self.roundNumber * round(problem[j][1] / float(self.roundNumber)),
                                 self.roundNumber * round(problem[j][2] / float(self.roundNumber)),
                                 self.roundNumber * round(problem[j][3] / float(self.roundNumber)))

        # Initialises the two arrays that will be returned
        permutedProblem = [(-1, -1, -1, -1)]*self.points
        permutation = [-1]*self.points

        # Iterates through the problem
        for j in range(0, self.points):
            lowestVal = (10000, 10000, 10000, 10000)
            lowestIndex = -1
            for k in range(0, self.points):
                if (roundedProblem[k] in permutedProblem):
                    # Point already in the permuted list so continue
                    continue
                else:
                    # Checks to see if this point is lower than the current lowest
                    if (roundedProblem[k][0] < lowestVal[0]):
                        lowestVal = roundedProblem[k]
                        lowestIndex = k
                    elif (roundedProblem[k][0] == lowestVal[0]):
                        if (roundedProblem[k][1] < lowestVal[1]):
                            lowestVal = roundedProblem[k]
                            lowestIndex = k
                        elif (roundedProblem[k][1] == lowestVal[1]):
                            if (roundedProblem[k][2] < lowestVal[2]):
                                lowestVal = roundedProblem[k]
                                lowestIndex = k
                            elif (roundedProblem[k][2] == lowestVal[2]):
                                if (roundedProblem[k][3] < lowestVal[3]):
                                    lowestVal = roundedProblem[k]
                                    lowestIndex = k

            # Adds the lowest point to the permuted list
            permutedProblem[j] = lowestVal
            permutation[j] = lowestIndex

        # Returns the new problem and index
        return permutedProblem, permutation

    # Updates all the arrays that are used in storing the action-space
    # Uses the Q-Learning algorithm to update them
    def __updateArrays(self, problem, previousOrder, action, success, reward):
        # Sets up the problem data
        problemData = str(problem + [previousOrder])

        try:
            # Gets the index of the problem
            problemIndex = self.problems[problemData]

            if (action in self.actions[problemIndex]):
                # If the problem and action has already been done, need to edit the reward
                self.reseenProblems += 1

                # Gets the index of the action
                # This doesn't need a try-catch as a linear search is faster
                # This is because the number of elements will always be very low
                actionIndex = self.actions[problemIndex].index(action)

                if (self.__getNextSuccess(str(problem + [previousOrder + [action]])) == 0):
                    # If all the next actions result in an unsuccessful ordering then set the reward to 0
                    self.rewards1[problemIndex][actionIndex] = 0
                    self.rewards2[problemIndex][actionIndex] = 0

                    # Updates the success value to be 0
                    self.success[problemIndex][actionIndex] = 0
                else:
                    # Updates the success value
                    self.success[problemIndex][actionIndex] = success

                    randInt = random.random()
                    if (randInt < 0.5):
                        # Gets the expected future reward
                        expectedFutureReward = self.__getExpectedFutureReward(str(problem + [previousOrder + [action]]), 1)

                        # Updates the first reward value using the Q-Learning algorithm
                        self.rewards1[problemIndex][actionIndex] = self.rewards1[problemIndex][actionIndex] + (self.alpha * (reward + (self.gamma * expectedFutureReward) - self.rewards1[problemIndex][actionIndex]))
                    else:
                        # Gets the expected future reward
                        expectedFutureReward = self.__getExpectedFutureReward(str(problem + [previousOrder + [action]]), 2)

                        # Updates the second reward value using the Q-Learning algorithm
                        self.rewards2[problemIndex][actionIndex] = self.rewards2[problemIndex][actionIndex] + (self.alpha * (reward + (self.gamma * expectedFutureReward) - self.rewards2[problemIndex][actionIndex]))
        except KeyError:
            # If it is a new problem then need to add the problem, action and reward

            # Adds the problem to the array and gets the index
            self.problems[problemData] = len(self.problems)
            problemIndex = self.problems[problemData]

            # Adds the arrays for the action, success and reward
            self.actions.append([])
            self.success.append([])
            self.rewards1.append([])
            self.rewards2.append([])

            # Adds all the possible actions for this problem
            # Initialises the success as 1 and the reward as 0 if it hasn't been seen yet
            for nextAction in range(0, self.points):
                if (nextAction in previousOrder):
                    # Action already done so can't do it again and doesn't need to be added to the arrays
                    pass
                else:
                    if (nextAction == action):
                        # Adds the action and success
                        self.actions[problemIndex].append(nextAction)
                        self.success[problemIndex].append(success)

                        randInt = random.random()
                        if (randInt < 0.5):
                            # Gets the expected future reward
                            expectedFutureReward = self.__getExpectedFutureReward(str(problem + [previousOrder + [nextAction]]), 1)

                            # Adds the reward to rewards 1 using the Q-Learning algorithm and previous reward as 0
                            self.rewards1[problemIndex].append(self.alpha * (reward + (self.gamma * expectedFutureReward)))
                            self.rewards2[problemIndex].append(0)
                        else:
                            # Gets the expected future reward
                            expectedFutureReward = self.__getExpectedFutureReward(str(problem + [previousOrder + [nextAction]]), 2)

                            # Adds the reward to rewards 2 using the Q-Learning algorithm and previous reward as 0
                            self.rewards2[problemIndex].append(self.alpha * (reward + (self.gamma * expectedFutureReward)))
                            self.rewards1[problemIndex].append(0)
                    else:
                        # Adds the actions not seen with success 1
                        self.actions[problemIndex].append(nextAction)
                        self.success[problemIndex].append(1)

                        # Adds a reward of 0
                        self.rewards1[problemIndex].append(0)
                        self.rewards2[problemIndex].append(0)

    # Gets the minimum reward of the new problem state
    def __getExpectedFutureReward(self, newProblemState, dictToUse):
        # Gets the expected future reward of the next actions
        try:
            # Gets the index of the new problem
            newProblemIndex = self.problems[newProblemState]

            if (dictToUse == 1):
                # Gets the maximum of the actions in the first array as this is the action that will be taken
                return max(self.rewards1[newProblemIndex])
            else:
                # Gets the maximum of the actions in the second array as this is the action that will be taken
                return max(self.rewards2[newProblemIndex])
        except KeyError:
            # Problem not seen before so return 0
            return 0

    def __getNextSuccess(self, newProblemState):
        # Gets whether the next state returns at least 1 successful ordering
        try:
            # Gets the index of the new problem
            newProblemIndex = self.problems[newProblemState]

            # Returns 1 if there is a successful ordering and 0 if all orderings are unsuccessful
            return max(self.success[newProblemIndex])
        except KeyError:
            # Problem not seen so presume success
            return 1

    def getNumberOfReseenProblems(self):
        # Returns the counter reseenProblems
        return self.reseenProblems
