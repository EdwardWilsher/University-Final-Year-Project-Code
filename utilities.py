import numpy
import math
import random

# region Constants

# Constant for the max reward value per point
MaxRewardPerPoint = 2000

# endregion

# region Check Valid Problem

# Checks whether the problem passed in is valid
# Does this by checking whether each point is at least 15 units away from each other point
def checkValidProblem(problem):
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

# endregion

# region Neighbours

# Gets the neighbours of the current solution
# For this a neighbour is defined as being an ordering with only 2 values swapped
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

# endregion

# region Generate Smaller Problem

# Initialises the problem array
def generateSmallerProblem(numberOfPoints, squareSize):
    problem = []

    # return [(480, 480, 1481, 481), (519, 479, 1520, 480), (500, 520, 1501, 521)]

    for point in range(0, numberOfPoints):
        problem.append((0, 0, 0, 0))

        validPoint = False
        while (validPoint is False):
            # Gets the first coordinate that's within the sqaure
            # point1 = random.randint(500 - (squareSize / 2), 500 + (squareSize / 2))
            # point2 = random.randint(500 - (squareSize / 2), 500 + (squareSize / 2))
            # point1 = random.randint(50 - (squareSize / 20), 50 + (squareSize / 20)) * 10
            # point2 = random.randint(50 - (squareSize / 20), 50 + (squareSize / 20)) * 10
            if (numberOfPoints == 3):
                if (point == 0):
                    point1 = random.randint(487 - squareSize, 487 + squareSize)
                    point2 = point1

                    point3 = random.randint(1487 - squareSize, 1487 + squareSize)
                    point4 = point3 - 1000
                elif (point == 1):
                    point1 = random.randint(513 - squareSize, 513 + squareSize)
                    point2 = point1 - 26

                    point3 = random.randint(1513 - squareSize, 1513 + squareSize)
                    point4 = point3 - 1026
                else:
                    point1 = random.randint(500 - squareSize, 500 + squareSize)
                    point2 = point1 + 13

                    point3 = random.randint(1500 - squareSize, 1500 + squareSize)
                    point4 = point3 - 987
            elif (numberOfPoints == 7):
                if (point == 0):
                    point1 = random.randint(500 - squareSize, 500 + squareSize)
                    point2 = point1 + 50

                    point3 = random.randint(1500 - squareSize, 1500 + squareSize)
                    point4 = point3 - 950
                elif (point == 1):
                    point1 = random.randint(530 - squareSize, 530 + squareSize)
                    point2 = point1 - 15

                    point3 = random.randint(1530 - squareSize, 1530 + squareSize)
                    point4 = point3 - 1015
                elif (point == 2):
                    point1 = random.randint(530 - squareSize, 530 + squareSize)
                    point2 = point1 - 50

                    point3 = random.randint(1530 - squareSize, 1530 + squareSize)
                    point4 = point3 - 1050
                elif (point == 3):
                    point1 = random.randint(510 - squareSize, 510 + squareSize)
                    point2 = point1 - 60

                    point3 = random.randint(1510 - squareSize, 1510 + squareSize)
                    point4 = point3 - 1060
                elif (point == 4):
                    point1 = random.randint(490 - squareSize, 490 + squareSize)
                    point2 = point1 - 40

                    point3 = random.randint(1490 - squareSize, 1490 + squareSize)
                    point4 = point3 - 1040
                elif (point == 5):
                    point1 = random.randint(470 - squareSize, 470 + squareSize)
                    point2 = point1 + 10

                    point3 = random.randint(1470 - squareSize, 1470 + squareSize)
                    point4 = point3 - 990
                elif (point == 6):
                    point1 = random.randint(470 - squareSize, 470 + squareSize)
                    point2 = point1 + 45

                    point3 = random.randint(1470 - squareSize, 1470 + squareSize)
                    point4 = point3 - 965

            # Gets the second coordiante that's within the square
            # point3 = random.randint(1500 - (squareSize / 2), 1500 + (squareSize / 2))
            # point4 = random.randint(500 - (squareSize / 2), 500 + (squareSize / 2))
            # point3 = random.randint(150 - (squareSize / 20), 150 + (squareSize / 20)) * 10
            # point4 = random.randint(50 - (squareSize / 20), 50 + (squareSize / 20)) * 10

            problem[point] = (point1, point2, point3, point4)

            # Checks that the point is 15 units away from all previous points
            validPoint = checkValidProblem(problem)

    # Returns the array
    return problem

# endregion

# region Output

# Outputs the percentage complete
# Also outputs the number of problems re-seen
def outputPercentageComplete(problemNumber, totalNumber, numberOfReseenProblems):
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

# endregion
