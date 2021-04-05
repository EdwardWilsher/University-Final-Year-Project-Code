import numpy
import random
import copt
import abc
import heuristics
import utilities
import threading

# region Base Class

# Base Reinforcement Learning Agent that forces all RL Agents to implement the methods learn and getSolution
class BaseRLAgent:
    # Declares this class as an abstract class
    __metaclass__ = abc.ABCMeta

    # This method will allow the RL agent to learn
    @abc.abstractmethod
    def learn(self, numberOfProblems, squareSize):
        return

    # This method will return the computed solution to the problem
    @abc.abstractmethod
    def getSolution(self, problem):
        return

# endregion

# region Q-Learning

# Q-Learning agent
class QLearningAgent(BaseRLAgent):

    # region Constructor

    # Constructor that initialises all the arrays and constants
    def __init__(self, numberOfPoints, numberToRound):
        # Initialises the Q-Learning dictionary/action space
        self.problems = {}
        self.actions = []
        self.success = []
        self.rewards1 = []
        self.rewards2 = []

        # Initialises two array that stores the uncertainty of the action
        self.actionUncertainty1 = []
        self.actionUncertainty2 = []

        # Initialises the number of points that need to be connected
        self.points = numberOfPoints

        # Initialises alpha and gamma which are used in the Q-Learning algorithm
        # Alpha is the learning rate and Gamma represents how much the agent values future rewards
        self.alpha = 0.1
        self.gamma = 0.8

        # Initialises epsilon, the chance a random action is chosen in action for the agent to learn
        self.epsilon = 0.1

        # Initialises the number to round to
        self.roundNumber = numberToRound

        # Initialises the reseen problem counter
        self.reseenProblems = 0

        # Initialises the arrays that will store the rewards and successes when the agent is learning
        self.learnRewards = []
        self.learnSuccesses = []

        # Initialises the minimum change and number of times seen variable
        self.minChange = 0.7
        self.requiredTimesSeen = 3

        # Initialises the counters and temporary reward arrays
        self.timesSeen1 = []
        self.timesSeen2 = []
        self.tempReward1 = []
        self.tempReward2 = []

    # endregion

    # region Learn

    # Takes in a problem and learns from the actions it takes in it
    def __learnProblem(self, learnProblem, index):
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
            reward = (utilities.MaxRewardPerPoint * len(newOrder)) - result["measure"]
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
                self.learnRewards[index] = 0
                self.learnSuccesses[index] = 0
                return

        # Gone through the for-loop so ordering is successful
        self.learnRewards[index] = reward
        self.learnSuccesses[index] = 1

    # Allows the agent to learn using the specified number of problems
    def learn(self, numberOfProblems, squareSize):
        print("Learning Started")

        if (numberOfProblems <= 0):
            print("Learning Finished")
            return

        for i in range(0, numberOfProblems):
            validProblem = False

            # Makes sure the problem is valid
            while (validProblem is False):
                # Get the problem
                problem = copt.getProblem(self.points) if squareSize == 0 else utilities.generateSmallerProblem(self.points, squareSize)

                # Checks whether the problem is valid
                validProblem = utilities.checkValidProblem(problem) if squareSize == 0 else True

            # Add an index to the reward and success arrays
            self.learnRewards.append(0)
            self.learnSuccesses.append(0)

            # Create and start the threads
            thread = threading.Thread(target=self.__learnProblem, args=(problem, len(self.learnRewards) - 1))
            thread.start()

            outputThread = threading.Thread(target=utilities.outputPercentageComplete, args=(i + 1, numberOfProblems, self.reseenProblems,))
            outputThread.start()

        while (thread.is_alive() or outputThread.is_alive()):
            # Waits for all threads to be completed
            pass

        print("Learning Finished")

    # Gets the next action when the agent is learning
    def __getNextLearnAction(self, problem, order):
        # Sets up the problem data to be how all problem data is stored
        problemData = str(problem + [order])

        # Checks to see if the problem has already been seen
        try:
            # Gets the index of the problem
            # This will be the index in all the other lists
            problemIndex = self.problems[problemData]

            # Sets up the arrays to store the optimal next action and optimal reward
            optimalNextAction = [-1]
            optimalReward = -10000

            # Iterates through all the tried actions
            for i in range(0, len(self.actions[problemIndex])):
                # Want to find the action that is a combination of the one that gives the best reward
                # and the one that will allow the agent to learn a lot
                # Once the agent has learnt for a while, expectedInformation ~= 0
                # So the best action is always chosen
                expectedReward = (self.rewards1[problemIndex][i] + self.rewards2[problemIndex][i]) / 2

                # Adds a ratio of the max possible reward depending on how uncertain the action is
                expectedInformation = (utilities.MaxRewardPerPoint * (len(order)+2)) * ((self.actionUncertainty1[problemIndex][i] + self.actionUncertainty2[problemIndex][i]) / 2)

                expectedReward = expectedReward + expectedInformation

                # Checks to see if the action was successful and checks that a higher reward was returned
                if ((self.success[problemIndex][i] == 1) and (expectedReward > optimalReward)):
                    optimalNextAction = [self.actions[problemIndex][i]]
                    optimalReward = expectedReward
                elif ((self.success[problemIndex][i] == 1) and (expectedReward == optimalReward)):
                    optimalNextAction.append(self.actions[problemIndex][i])

            # Checks if a valid ordering was found
            if (optimalNextAction == [-1]):
                # No optimal action found so just return a random one
                return heuristics.RandomHeuristic().getNextAction(problem, order)
            else:
                # Valid action found
                if (len(optimalNextAction) > 1):
                    # If there are multiple options, choose one at random
                    return random.choice(optimalNextAction)
                else:
                    # Only 1 optimal action so return it
                    return optimalNextAction[0]
        except KeyError:
            # Problem not seen already so return a random action
            return heuristics.RandomHeuristic().getNextAction(problem, order)

    # endregion

    # region Get Solution

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
            reward = (utilities.MaxRewardPerPoint * len(newOrder)) - result["measure"]
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

    # endregion

    # region Permute

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

    # endregion

    # region Update Q-Table

    # Updates all the arrays that are used in storing the action-space
    # Uses the Q-Learning algorithm to update them
    def __updateArrays(self, problem, previousOrder, action, success, reward):
        # Sets up the problem data
        problemData = str(problem + [previousOrder])

        try:
            # Gets the index of the problem
            problemIndex = self.problems[problemData]

            # Increment a reseen problems counter by 1
            self.reseenProblems += 1
        except KeyError:
            # If it is a new problem then need to add 0 values to all the arrays
            self.__addActions(problemData, previousOrder)

            # Gets the index of the problem
            problemIndex = self.problems[problemData]

        # Gets the index of the action
        actionIndex = self.actions[problemIndex].index(action)

        if ((self.__getNextSuccess(str(problem + [previousOrder + [action]])) == 0) or (success == 0)):
            # If all the next actions result in an unsuccessful ordering then set the reward to 0
            self.rewards1[problemIndex][actionIndex] = 0
            self.rewards2[problemIndex][actionIndex] = 0

            # Updates the success value to be 0
            self.success[problemIndex][actionIndex] = 0

            # Updates the uncertainty to 0
            self.actionUncertainty1[problemIndex][actionIndex] = 0
            self.actionUncertainty2[problemIndex][actionIndex] = 0
        else:
            # Updates the success value
            self.success[problemIndex][actionIndex] = success

            randInt = random.random()
            if (randInt < 0.5):
                # Gets the expected future reward
                expectedFutureReward = self.__getExpectedFutureReward(
                    str(problem + [previousOrder + [action]]), 1)

                # Update the temp reward and counter
                self.tempReward1[problemIndex][actionIndex] = self.tempReward1[problemIndex][actionIndex] + reward + (self.gamma * expectedFutureReward)
                self.timesSeen1[problemIndex][actionIndex] += 1

                # Checks to see if the problem and action has been seen enough
                if (self.timesSeen1[problemIndex][actionIndex] == self.requiredTimesSeen):
                    # Check to see if there has been enough of a change
                    if (abs(self.rewards1[problemIndex][actionIndex] - (self.tempReward1[problemIndex][actionIndex] / self.requiredTimesSeen)) >= (2 * self.minChange)):
                        # Change the reward
                        self.rewards1[problemIndex][actionIndex] = (self.tempReward1[problemIndex][actionIndex] / self.requiredTimesSeen) + self.minChange

                    # Reset the variables to 0
                    self.tempReward1[problemIndex][actionIndex] = 0
                    self.timesSeen1[problemIndex][actionIndex] = 0

                    # Update the uncertainty of the values
                    if ((len(previousOrder) + 1) == self.points):
                        # Update the uncertainty to 0
                        self.actionUncertainty1[problemIndex][actionIndex] = 0
                    else:
                        # Update the uncertainty of the action
                        self.actionUncertainty1[problemIndex][actionIndex] = self.__getUncertainty(problem, previousOrder, action, 1)
            else:
                # Gets the expected future reward
                expectedFutureReward = self.__getExpectedFutureReward(str(problem + [previousOrder + [action]]), 2)

                # Update the temp reward and counter
                self.tempReward2[problemIndex][actionIndex] = self.tempReward2[problemIndex][actionIndex] + reward + (self.gamma * expectedFutureReward)
                self.timesSeen2[problemIndex][actionIndex] += 1

                # Checks to see if the problem and action has been seen enough
                if (self.timesSeen2[problemIndex][actionIndex] == self.requiredTimesSeen):
                    # Check to see if there has been enough of a change
                    if (abs(self.rewards2[problemIndex][actionIndex] - (self.tempReward2[problemIndex][actionIndex] / self.requiredTimesSeen)) >= (2 * self.minChange)):
                        # Change the reward
                        self.rewards2[problemIndex][actionIndex] = (self.tempReward2[problemIndex][actionIndex] / self.requiredTimesSeen) + self.minChange

                    # Reset the variables to 0
                    self.tempReward2[problemIndex][actionIndex] = 0
                    self.timesSeen2[problemIndex][actionIndex] = 0

                    # Update the uncertainty of the values
                    if ((len(previousOrder) + 1) == self.points):
                        # Update the uncertainty to 0
                        self.actionUncertainty2[problemIndex][actionIndex] = 0
                    else:
                        # Update the uncertainty of the action
                        self.actionUncertainty2[problemIndex][actionIndex] = self.__getUncertainty(problem, previousOrder, action, 2)

    # Adds all possible actions to the arrays
    def __addActions(self, problemData, previousOrder):
        # If it is a new problem then need to add 0 values to all the arrays

        # Adds the problem to the array and gets the index
        self.problems[problemData] = len(self.problems)
        problemIndex = self.problems[problemData]

        # Adds the arrays for the action, success, rewards, uncertainty, temporary rewards and counters
        self.actions.append([])
        self.success.append([])
        self.rewards1.append([])
        self.rewards2.append([])
        self.actionUncertainty1.append([])
        self.actionUncertainty2.append([])
        self.tempReward1.append([])
        self.tempReward2.append([])
        self.timesSeen1.append([])
        self.timesSeen2.append([])

        # Adds all the possible actions for this problem
        # Initialises the success as 1 and the reward as 0 if it hasn't been seen yet
        for nextAction in range(0, self.points):
            if (nextAction in previousOrder):
                # Action already done so can't do it again and doesn't need to be added to the arrays
                pass
            else:
                # Adds the actions not seen with success 1
                self.actions[problemIndex].append(nextAction)
                self.success[problemIndex].append(1)

                # Adds a reward of 0
                self.rewards1[problemIndex].append(0)
                self.rewards2[problemIndex].append(0)

                # Adds an uncertainty of 1
                self.actionUncertainty1[problemIndex].append(1)
                self.actionUncertainty2[problemIndex].append(1)

                # Adds 0 to the temp rewards and counters
                self.tempReward1[problemIndex].append(0)
                self.tempReward2[problemIndex].append(0)
                self.timesSeen1[problemIndex].append(0)
                self.timesSeen2[problemIndex].append(0)

    # endregion

    # region Private Getters

    # Gets the expected future reward of the next actions
    def __getExpectedFutureReward(self, newProblemState, dictToUse):
        try:
            # Gets the index of the new problem
            newProblemIndex = self.problems[newProblemState]

            if (dictToUse == 1):
                # Gets the action which returns the highest reward in the first array
                actionIndex = self.rewards2[newProblemIndex].index(max(self.rewards2[newProblemIndex]))

                # Return the predicted reward of the action in the second array
                return self.rewards1[newProblemIndex][actionIndex]
            else:
                # Gets the action which returns the highest reward in the second array
                actionIndex = self.rewards1[newProblemIndex].index(max(self.rewards1[newProblemIndex]))

                # Return the predicted reward of the action in the first array
                return self.rewards2[newProblemIndex][actionIndex]
        except KeyError:
            # Problem not seen before so return 0
            return 0

    # Gets whether the next state returns at least 1 successful ordering
    def __getNextSuccess(self, newProblemState):
        try:
            # Gets the index of the new problem
            newProblemIndex = self.problems[newProblemState]

            # Returns 1 if there is a successful ordering and 0 if all orderings are unsuccessful
            return max(self.success[newProblemIndex])
        except KeyError:
            # Problem not seen so presume success
            return 1

    # Returns the uncertainty of doing the action in the current problem
    def __getUncertainty(self, problem, order, action, arrayToUse):
        # Get the problem
        problemData = str(problem + [order + [action]])

        try:
            # Gets the index of the problem
            problemIndex = self.problems[problemData]

            # Gets the array to use
            uncertaintyArray = self.actionUncertainty1[problemIndex] if (arrayToUse == 1) else self.actionUncertainty2[problemIndex]

            # Returns the average of the future uncertainties
            return (sum(uncertaintyArray) / float(len(uncertaintyArray)))
        except KeyError:
            # Problem not seen so uncertainty is 1
            return 1

    # endregion

    # region Public Getters

    # Returns the counter reseenProblems
    def getNumberOfReseenProblems(self):
        return self.reseenProblems

    # Returns the learning rewards and successes
    def getLearnValues(self):
        return self.learnSuccesses, self.learnRewards

    # endregion

# endregion
