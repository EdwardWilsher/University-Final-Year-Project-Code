import numpy
import copt
import abc
import random
import math
import utilities

# region Base Class

# Base Heuristic class for all the Combinatorial Optimisation Heuristics
# These Heuristics will be used to provide a guide as to how well the RL agent is doing
class BaseHeuristic():
    # Declares this class as an abstract class
    __metaclass__ = abc.ABCMeta

    # Each Sub-Class needs a method called getSolution
    @abc.abstractmethod
    def getSolution(self, problem):
        return

# endregion

# region Manhattan Heuristic

# This class will be used to work out the solution of the problem using the Manhattan Heuristic
class ManhattanHeuristic(BaseHeuristic):

    # region Get Solution

    # Gets a solution to the problem using the Manhattan Heuristic
    def getSolution(self, problem):
        # Sets up the order and distance variables
        order = []
        distances = []

        # Gets the number of points in the problem
        numberOfPoints = len(problem)

        for x in range(0, numberOfPoints):
            # Calculates the distance of the point
            distance = math.sqrt(pow((problem[x][0] - problem[x][2]), 2) + pow((problem[x][1] - problem[x][3]), 2))

            # Flag to indicate if the index has been inserted into the order
            # If it hasn't, it needs to be added to the end
            inserted = False

            # Iterates through all the distances currently in the array
            maxIndex = len(distances)
            for y in range(0, maxIndex):
                # Makes sure that distances is an ordered list
                if (distance < distances[y]):
                    distances.insert(y, distance)
                    order.insert(y, x)
                    inserted = True
                    break

            # If the distance wasn't inserted then it must be the highest value and need to go at the end
            if (inserted is False):
                distances.append(distance)
                order.append(x)

        # Submits the ordering to see how well it did and returns the result and action
        result = copt.evaluate(problem, order)
        return result['order'], result['success'], (utilities.MaxRewardPerPoint * numberOfPoints) - result['measure']

    # endregion

    # region Get Next Action

    # Finds the next point to connect when using the manhattan heuristic
    @staticmethod
    def getNextAction(problem, previousOrder):
        # Sets up the variables
        nextAction = -1
        shortestDistance = 100000

        # Gets the number of points in the problem
        numberOfPoints = len(problem)

        for x in range(0, numberOfPoints):
            if (x in previousOrder):
                # x already connected so need to ignore it
                continue
            else:
                # Calculates the distance of the point
                distance = math.sqrt(pow((problem[x][0] - problem[x][2]), 2) + pow((problem[x][1] - problem[x][3]), 2))

                if (distance < shortestDistance):
                    # If the distance is less than the previous distance
                    # Then need to store this action as it could be the best action to do
                    shortestDistance = distance
                    nextAction = x

        # Return the best action using the manhattan heuristic
        return nextAction

    # endregion

# endregion

# region Random Heuristic

# This class will be used to work out the solution of the problem using a random ordering
class RandomHeuristic(BaseHeuristic):

    # region Get Solution

    # Uses a random ordering to get a solution to the problem
    def getSolution(self, problem):
        # Sets up the order variable
        order = []

        # Gets the number of points in the problem
        numberOfPoints = len(problem)
        indexes = list(range(0, numberOfPoints))

        # Picks a random ordering
        for y in range(0, numberOfPoints):
            index = random.randint(0, numberOfPoints-y-1)
            order.append(indexes[index])
            indexes.remove(indexes[index])

        # Submits the ordering to see how well it did and returns the result and success
        result = copt.evaluate(problem, order)
        return result['order'], result['success'], (utilities.MaxRewardPerPoint * numberOfPoints) - result['measure']

    # endregion

    #region Get Next Action

    # Finds the next point to connect when using the random ordering
    @staticmethod
    def getNextAction(problem, previousOrder):
        # Gets the number of points in the problem
        numberOfPoints = len(problem)
        actions = list(range(0, numberOfPoints))

        # Gets the remaining options
        remainingActions = [x for x in actions if x not in previousOrder]

        # Returns a random action
        return random.choice(remainingActions)

    # endregion

# endregion

# region Hill Climbing Heuristic

# This class uses the Hill Climbing Heuristic to obtain solutions
class HillClimbing(BaseHeuristic):

    # Gets the solution via the hill climbing heuristic
    def getSolution(self, problem):
        # Get an initial random solution
        rand = RandomHeuristic()
        currentSolution, success, reward = rand.getSolution(problem)

        # Get the reward for it
        currentBestReward = 0 if success == 0 else reward

        # Loop while the neighbouring rewards are better
        bestNeighbourReward = currentBestReward + 0.1
        while (bestNeighbourReward > currentBestReward):
            currentBestReward = bestNeighbourReward
            bestNeighbourReward = 0
            neighbours = utilities.getNeighbours(currentSolution)

            # Loop through all the neighbours
            for neighbour in neighbours:
                # Get the result for this ordering
                result = copt.evaluate(problem, neighbour)

                # Check to see if this reward is better
                if ((result['success'] == 1) and (((utilities.MaxRewardPerPoint * len(problem)) - result['measure']) > bestNeighbourReward)):
                    bestNeighbourReward = (utilities.MaxRewardPerPoint * len(problem)) - result['measure']
                    currentSolution = neighbour

        # Gets all the values for the chosen ordering
        result = copt.evaluate(problem, currentSolution)
        return result['order'], result['success'], (utilities.MaxRewardPerPoint * len(problem)) - result['measure']

# endregion

# region Simulated Annealing Heuristic

# This class uses the simulated annealing Heuristic to obtain solutions
class SimulatedAnnealing(BaseHeuristic):

    # Gets the solution via simulated annealing
    def getSolution(self, problem):
        # Get an initial random solution
        rand = RandomHeuristic()
        currentSolution, success, reward = rand.getSolution(problem)

        # Get the reward for it
        currentBestReward = 0 if success == 0 else reward

        # Sets up the variables temperature and iterations and the constants alpha and beta
        temperature = 500
        iterations = 10
        alpha = 0.2
        beta = 2

        # Loop while temperature is greater than 1
        while (temperature > 1):
            for index in range(0, iterations):
                # Gets a neighbouring solution and the result
                neighbourSol = utilities.getPossibleNeighbourSolution(currentSolution)
                result = copt.evaluate(problem, neighbourSol)

                # Gets the reward
                newReward = 0 if result['success'] == 0 else (utilities.MaxRewardPerPoint * len(problem)) - result['measure']

                # Calculates the values used in the if-statement
                randomNum = random.random()
                expVal = math.exp((currentBestReward - newReward) / temperature)

                # Check to see if this reward is better or a random number is less than an exponential
                if ((newReward > currentBestReward) or (randomNum < expVal)):
                    currentSolution = neighbourSol
                    currentBestReward = newReward

            temperature = temperature * alpha
            iterations = iterations * beta

        # Gets all the values for the chosen ordering
        result = copt.evaluate(problem, currentSolution)
        return result['order'], result['success'], (utilities.MaxRewardPerPoint * len(problem)) - result['measure']

# endregion
