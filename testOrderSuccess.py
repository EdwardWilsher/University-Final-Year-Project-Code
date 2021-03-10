import numpy
import copt
import matplotlib.pyplot as pyplot
import heuristics
import rlAgents
import time
import utilities
import threading

# Number of problems that will be used to test the RL Agent and the number of points in each problem
numberOfProblems = int(input("Please input the number of problems to use to test the solutions: "))
numberOfPoints = int(input("Please input the number of points to connect: "))

# Allow the user to input the number of problems the agent will learn with
numberToLearn = int(input("Please input the number of problems the RL Agent will learn with: "))

# Allows the user to reduce the state-space by rounding
check = input("Do you want to reduce the state-space by rounding the problem (Y/N): ")

# If the user wants to reduce the state-space, need to know what number to round to
if (check.lower() == "y"):
    roundNum = int(input("What number do you want to round the problem coordinates: "))
else:
    roundNum = 1

# Initialise the Q-Learning agent
qLearningAgent = rlAgents.QLearningAgent(numberOfPoints, roundNum)

# Allow the user to reduce the square that the problem coordinates have to be in
check = input("Do you want to test on a smaller problem (Y/N): ")

# If the user wants to reduce the problem space, need to know the new square side length
if (check.lower() == "y"):
    squareSize = int(input("What square side length do you want: "))
else:
    squareSize = 0

# Start the timer
startTime = time.time()

# Allow the agent to learn
qLearningAgent.learn(numberToLearn, squareSize)

# Create the instances of the heuristics that will be used to test how good the RL Agent is
manhat = heuristics.ManhattanHeuristic()
rand = heuristics.RandomHeuristic()
hillClimb = heuristics.HillClimbing()
simAnneal = heuristics.SimulatedAnnealing()

# Stores the number of success for the heuristics and RL Agent
manhatSuccesses = 0
randSuccesses = 0
hillClimbSuccesses = 0
simAnnealSuccesses = 0
qLearnSuccesses = 0

def getSolutions(givenProblem):
    # Define all the variables as global
    global manhatSuccesses
    global randSuccesses
    global hillClimbSuccesses
    global simAnnealSuccesses
    global qLearnSuccesses

    # Gets the heuristic solutions to the problem
    _, manhatSuccess, _ = manhat.getSolution(givenProblem)
    _, randSuccess, _ = rand.getSolution(givenProblem)
    _, hillClimbSuccess, _ = hillClimb.getSolution(givenProblem)
    _, simAnnealSuccess, _ = simAnneal.getSolution(givenProblem)

    # Gets the RL Agent solution to the problem
    _, qLearnSuccess, _ = qLearningAgent.getSolution(givenProblem)

    # Change the counters with a lock to prevent multiple threads changing the values at the same time
    threading.Lock().acquire()
    try:
        manhatSuccesses += manhatSuccess
        randSuccesses += randSuccess
        hillClimbSuccesses += hillClimbSuccess
        simAnnealSuccesses += simAnnealSuccess
        qLearnSuccesses += qLearnSuccess

        threading.Lock().release()
    except RuntimeError:
        pass

    return


print("Testing Started")

for x in range(0, numberOfProblems):
    validProblem = False

    while (validProblem is False):
        # Get the random problem
        problem = copt.getProblem(numberOfPoints) if squareSize == 0 else utilities.generateSmallerProblem(numberOfPoints, squareSize)

        # Checks whether the problem is valid
        validProblem = utilities.checkValidProblem(problem) if squareSize == 0 else True

    # Gets the solutions to the problem in a separate thread
    thread = threading.Thread(target=getSolutions, args=(problem,))
    thread.start()

    outputThread = threading.Thread(target=utilities.outputPercentageComplete, args=(x + 1, numberOfProblems, qLearningAgent.getNumberOfReseenProblems(),))
    outputThread.start()

while (thread.is_alive() or outputThread.is_alive()):
    # Waits for all threads to be completed
    pass

print("Testing Finished")

# Stop the timer
timeTaken = time.time() - startTime
print(timeTaken)

# Get the x-axis (number of different heuristics) and y-axis (percentage correct)
xAxis = [1, 2, 3, 4, 5]
percentageCorrect = [0, 0, 0, 0, 0]

# Calculate the percentage correct
percentageCorrect[0] = (manhatSuccesses / numberOfProblems) * 100
percentageCorrect[1] = (randSuccesses / numberOfProblems) * 100
percentageCorrect[2] = (hillClimbSuccesses / numberOfProblems) * 100
percentageCorrect[3] = (simAnnealSuccesses / numberOfProblems) * 100
percentageCorrect[4] = (qLearnSuccesses / numberOfProblems) * 100

print("Manhattan: " + str(percentageCorrect[0]))
print("Random: " + str(percentageCorrect[1]))
print("Hill Climbing: " + str(percentageCorrect[2]))
print("Simulated Annealing: " + str(percentageCorrect[3]))
print("Q-Learning: " + str(percentageCorrect[4]))

# Plots the bar chart
pyplot.bar(xAxis, percentageCorrect, tick_label = ["Manhattan", "Random", "Hill Climbing", "Simulated Annealing", "Q-Learning"], width = 0.8)

# Name the x-axis, y-axis and Title
pyplot.xlabel("Ordering Heuristic")
pyplot.ylabel("Percentage of Successful Orderings")
pyplot.title("Graph to show the success of the heuristics and the RL Agent")

# Show the plots
pyplot.show()
