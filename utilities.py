import numpy
import math
import random

# Constant for the max reward value per point
MaxRewardPerPoint = 2000

def checkValidProblem(problem):
    # Checks whether the problem passed in is valid
    # Does this by checking whether each point is at least 15 units away from each other point

    # Gets the number of points
    numberOfPoints = len(problem)

    for index1 in range(0, numberOfPoints):
        for index2 in range(0, numberOfPoints):
            if (index1 != index2):
                # Gets the distance between the points
                firstPointLength = math.sqrt((abs(problem[index1][0] - problem[index2][0]) ** 2) + (abs(problem[index1][1] - problem[index2][1]) ** 2))
                secondPointLength = math.sqrt((abs(problem[index1][2] - problem[index2][2]) ** 2) + (abs(problem[index1][3] - problem[index2][3]) ** 2))

                # Checks whether the points are 15 units apart
                if ((firstPointLength < 15) or (secondPointLength < 15)):
                    return False

    # All problems checked and are valid so return true
    return True

# Gets the neighbours of the current solution
# For this I define a neighbour as being an ordering with only 2 values swapped
def getNeighbours(currentSolution):
    # List of all the neighbour orderings
    neighbourOrderings = []

    # Iterate through all the indexes
    for firstIndex in range(0, len(currentSolution)):
        # Iterate through the points after it
        for secondIndex in range(firstIndex + 1, len(currentSolution)):
            # Swaps the points
            newOrder = list(currentSolution)
            temp = currentSolution[firstIndex]
            newOrder[firstIndex] = newOrder[secondIndex]
            newOrder[secondIndex] = temp

            # Add the ordering to the list
            neighbourOrderings.append(newOrder)

    # Returns the list
    return neighbourOrderings

# Gets a random neighbouring solution
def getPossibleNeighbourSolution(currentSolution):
    # Gets two random indexes
    firstIndex = random.randint(0, len(currentSolution) - 1)
    secondIndex = random.randint(0, len(currentSolution) - 2)
    secondIndex = secondIndex + 1 if secondIndex >= firstIndex else secondIndex

    # Swap the two values
    newOrder = list(currentSolution)
    temp = currentSolution[firstIndex]
    currentSolution[firstIndex] = currentSolution[secondIndex]
    currentSolution[secondIndex] = temp

    # Return the new order
    return newOrder

def generateSmallerProblem(numberOfPoints, squareSize):
    # Initialises the problem array
    problem = []

    for point in range(0, numberOfPoints):
        problem.append((0, 0, 0, 0))

        validPoint = False
        while (validPoint is False):
            # Gets the first coordinate that's within the sqaure
            #point1 = random.randint(500 - (squareSize / 2), 500 + (squareSize / 2))
            #point2 = random.randint(500 - (squareSize / 2), 500 + (squareSize / 2))
            point1 = random.randint(50 - (squareSize / 20), 50 + (squareSize / 20)) * 10
            point2 = random.randint(50 - (squareSize / 20), 50 + (squareSize / 20)) * 10

            # Gets the second coordiante that's within the square
            #point3 = random.randint(1500 - (squareSize / 2), 1500 + (squareSize / 2))
            #point4 = random.randint(500 - (squareSize / 2), 500 + (squareSize / 2))
            point3 = random.randint(150 - (squareSize / 20), 150 + (squareSize / 20)) * 10
            point4 = random.randint(50 - (squareSize / 20), 50 + (squareSize / 20)) * 10

            problem[point] = (point1, point2, point3, point4)

            # Checks that the point is 15 units away from all previous points
            validPoint = checkValidProblem(problem)

    # Returns the array
    return problem

def outputPercentageComplete(problemNumber, totalNumber, numberOfReseenProblems):
    # Outputs the percentage complete
    # Also outputs the number of problems re-seen
    if (((problemNumber / totalNumber) * 100) == 10):
        print("10% complete. Problems Reseen: ", numberOfReseenProblems)
    elif (((problemNumber / totalNumber) * 100) == 20):
        print("20% complete. Problems Reseen: ", numberOfReseenProblems)
    elif (((problemNumber / totalNumber) * 100) == 30):
        print("30% complete. Problems Reseen: ", numberOfReseenProblems)
    elif (((problemNumber / totalNumber) * 100) == 40):
        print("40% complete. Problems Reseen: ", numberOfReseenProblems)
    elif (((problemNumber / totalNumber) * 100) == 50):
        print("50% complete. Problems Reseen: ", numberOfReseenProblems)
    elif (((problemNumber / totalNumber) * 100) == 60):
        print("60% complete. Problems Reseen: ", numberOfReseenProblems)
    elif (((problemNumber / totalNumber) * 100) == 70):
        print("70% complete. Problems Reseen: ", numberOfReseenProblems)
    elif (((problemNumber / totalNumber) * 100) == 80):
        print("80% complete. Problems Reseen: ", numberOfReseenProblems)
    elif (((problemNumber / totalNumber) * 100) == 90):
        print("90% complete. Problems Reseen: ", numberOfReseenProblems)
    elif (((problemNumber / totalNumber) * 100) == 100):
        print("100% complete. Problems Reseen: ", numberOfReseenProblems)
