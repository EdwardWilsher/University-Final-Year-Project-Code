import numpy
import copt
import utilities

# Gets the best solution for the inputted problem
def getSolution(problem):
    # Gets all the solutions
    solutions = copt.bruteForce(problem)

    # The best solution is always the first one
    if (len(solutions) != 0):
        bestOrder = solutions[0]['order']
        bestReward = (utilities.MaxRewardPerPoint * len(problem)) - solutions[0]['measure']
        success = solutions[0]['success']

        return bestOrder, success, bestReward
    else:
        return list(range(0, len(problem))), 0, 0
