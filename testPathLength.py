import numpy
import copt
import matplotlib.pyplot as pyplot
import bruteForce
import heuristics
import rlAgents
import time
import utilities
import threading

# region User Input

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

# Allow the user to reduce the square that the problem coordinates have to be in
check = input("Do you want to test on a smaller problem (Y/N): ")

# If the user wants to reduce the problem space, need to know the new square side length
if (check.lower() == "y"):
    squareSize = int(input("What square side length do you want: "))
else:
    squareSize = 0

# endregion

# region Learning

# Initialise the Q-Learning agent
qLearningAgent = rlAgents.QLearningAgent(numberOfPoints, roundNum)

# Gets the start time
startTime = time.time()

# Allow the agent to learn
qLearningAgent.learn(numberToLearn, squareSize)

# Gets the time after the learning has finished and calculates how long the learning took
timeTaken = time.time() - startTime
print("Learning took", timeTaken, "seconds")

# endregion

# region Testing

# Gets the mid time
midTime = time.time()

# Create the instances of the heuristics that will be used to test how good the RL Agent is
manhat = heuristics.ManhattanHeuristic()
rand = heuristics.RandomHeuristic()
hillClimb = heuristics.HillClimbing()
simAnneal = heuristics.SimulatedAnnealing()

# Arrays that will be used to store the orderings, success and solutions
# For the heuristics, the RL agent and the brute force approach
manhatOrders = [[-1] * numberOfPoints] * numberOfProblems
manhatSuccesses = [-1] * numberOfProblems
manhatRewards = [-1] * numberOfProblems

randOrders = [[-1] * numberOfPoints] * numberOfProblems
randSuccesses = [-1] * numberOfProblems
randRewards = [-1] * numberOfProblems

hillClimbOrders = [[-1] * numberOfPoints] * numberOfProblems
hillClimbSuccesses = [-1] * numberOfProblems
hillClimbRewards = [-1] * numberOfProblems

simAnnealOrders = [[-1] * numberOfPoints] * numberOfProblems
simAnnealSuccesses = [-1] * numberOfProblems
simAnnealRewards = [-1] * numberOfProblems

qLearnOrders = [[-1] * numberOfPoints] * numberOfProblems
qLearnSuccesses = [-1] * numberOfProblems
qLearnRewards = [-1] * numberOfProblems

bruteOrders = [[-1] * numberOfPoints] * numberOfProblems
bruteSuccesses = [-1] * numberOfProblems
bruteRewards = [-1] * numberOfProblems

print("Testing Started")

for x in range(0, numberOfProblems):
    validProblem = False

    while (validProblem is False):
        # Get the random problem
        problem = copt.getProblem(numberOfPoints) if squareSize == 0 else utilities.generateSmallerProblem(
            numberOfPoints, squareSize)

        # Checks whether the problem is valid
        validProblem = utilities.checkValidProblem(problem) if squareSize == 0 else True

        if (validProblem is True):
            # Gets the brute force solution to the problem
            # This is used a baseline to see how close to the correct solution the RL Agent is
            bruteOrders[x], bruteSuccesses[x], bruteRewards[x] = bruteForce.getSolution(problem)

            # If the brute force was unsuccessful then ignore the problem
            validProblem = False if bruteSuccesses[x] == 0 else True

    # Gets the heuristic solutions to the problem
    manhatOrders[x], manhatSuccesses[x], manhatRewards[x] = manhat.getSolution(problem)
    randOrders[x], randSuccesses[x], randRewards[x] = rand.getSolution(problem)
    hillClimbOrders[x], hillClimbSuccesses[x], hillClimbRewards[x] = hillClimb.getSolution(problem)
    simAnnealOrders[x], simAnnealSuccesses[x], simAnnealRewards[x] = simAnneal.getSolution(problem)

    # Gets the RL Agent solution to the problem
    qLearnOrders[x], qLearnSuccesses[x], qLearnRewards[x] = qLearningAgent.getSolution(problem)

    # Starts the thread to output the percentage complete
    outputThread = threading.Thread(target=utilities.outputPercentageComplete,
                                    args=(x + 1, numberOfProblems, qLearningAgent.getNumberOfReseenProblems(),))
    outputThread.start()

while (outputThread.is_alive()):
    # Waits for all threads to be completed
    pass

print("Testing Finished")

# Gets the time after the testing has finished and calculates how long the testing took
timeTaken = time.time() - midTime
print("Testing took", timeTaken, "seconds")

# endregion

# region Formatting

# Lists of the rewards for the successful orderings
manhatRewSuc = []
randRewSuc = []
hillClimbRewSuc = []
simAnnealRewSuc = []
qLearnRewSuc = []

# Iterates through the arrays to smooth them
# If the heuristic or agent was unsuccessful then that needs to be smoothed over
# Otherwise it is added to the list of successful rewards
for x in range(0, numberOfProblems):
    if (manhatSuccesses[x] == 0):
        manhatRewards[x] = bruteRewards[x] - 300
    else:
        manhatRewSuc.append(bruteRewards[x] - manhatRewards[x])

    if (randSuccesses[x] == 0):
        randRewards[x] = bruteRewards[x] - 300
    else:
        randRewSuc.append(bruteRewards[x] - randRewards[x])

    if (hillClimbSuccesses[x] == 0):
        hillClimbRewards[x] = bruteRewards[x] - 300
    else:
        hillClimbRewSuc.append(bruteRewards[x] - hillClimbRewards[x])

    if (simAnnealSuccesses[x] == 0):
        simAnnealRewards[x] = bruteRewards[x] - 300
    else:
        simAnnealRewSuc.append(bruteRewards[x] - simAnnealRewards[x])

    if (qLearnSuccesses[x] == 0):
        qLearnRewards[x] = bruteRewards[x] - 300
    else:
        qLearnRewSuc.append(bruteRewards[x] - qLearnRewards[x])

# Lists of the average rewards every 10 problems for the successful orderings
manhatAvRew = []
randAvRew = []
hillClimbAvRew = []
simAnnealAvRew = []
qLearnAvRew = []

# Lists of the temp rewards every 10 problems for the successful orderings
manhatTempRew = []
randTempRew = []
hillClimbTempRew = []
simAnnealTempRew = []
qLearnTempRew = []

# Iterates through the array to get the average reward every 10 problems
for x in range(0, numberOfProblems):
    # Add the current index
    manhatTempRew.append(bruteRewards[x] - manhatRewards[x])
    randTempRew.append(bruteRewards[x] - randRewards[x])
    hillClimbTempRew.append(bruteRewards[x] - hillClimbRewards[x])
    simAnnealTempRew.append(bruteRewards[x] - simAnnealRewards[x])
    qLearnTempRew.append(bruteRewards[x] - qLearnRewards[x])

    if (((x+1) % 10) == 0):
        # Append the averages
        manhatAvRew.append(sum(manhatTempRew) / len(manhatTempRew))
        randAvRew.append(sum(randTempRew) / len(randTempRew))
        hillClimbAvRew.append(sum(hillClimbTempRew) / len(hillClimbTempRew))
        simAnnealAvRew.append(sum(simAnnealTempRew) / len(simAnnealTempRew))
        qLearnAvRew.append(sum(qLearnTempRew) / len(qLearnTempRew))

        # Reset the temp lists
        manhatTempRew = []
        randTempRew = []
        hillClimbTempRew = []
        simAnnealTempRew = []
        qLearnTempRew = []

# endregion

# region Output

# Gets the end time and calculates how long the program took
timeTaken = time.time() - startTime
print("The total time that the program took, in seconds, was", timeTaken)

# Get the x-axis
xAxis = list(range(1, len(manhatAvRew) + 1))
scaledXAxis = [element*10 for element in xAxis]

fig, plots = pyplot.subplots(5, sharex=True)

# Plot the heuristics
plots[0].plot(scaledXAxis, manhatAvRew)
plots[1].plot(scaledXAxis, randAvRew)
plots[2].plot(scaledXAxis, hillClimbAvRew)
plots[3].plot(scaledXAxis, simAnnealAvRew)

# Plot the RL Agent
plots[4].plot(scaledXAxis, qLearnAvRew)

# Name the x-axis, y-axis and Title
plots[0].set(ylabel="Order Distance")
plots[1].set(ylabel="Order Distance")
plots[2].set(ylabel="Order Distance")
plots[3].set(ylabel="Order Distance")
plots[4].set(xlabel="Problem Number", ylabel="Order Distance")
plots[0].set_title("Manhattan Ordering")
plots[1].set_title("Random Ordering")
plots[2].set_title("Hill Climbing")
plots[3].set_title("Simulated Annealing")
plots[4].set_title("Q-Learning Agent")

# Checks to see if there was any orderings in the successful rewards lists
if (len(manhatRewSuc) == 0):
    manhatRewSuc.append(utilities.MaxRewardPerPoint * numberOfPoints)
if (len(randRewSuc) == 0):
    randRewSuc.append(utilities.MaxRewardPerPoint * numberOfPoints)
if (len(hillClimbRewSuc) == 0):
    hillClimbRewSuc.append(utilities.MaxRewardPerPoint * numberOfPoints)
if (len(simAnnealRewSuc) == 0):
    simAnnealRewSuc.append(utilities.MaxRewardPerPoint * numberOfPoints)
if (len(qLearnRewSuc) == 0):
    qLearnRewSuc.append(utilities.MaxRewardPerPoint * numberOfPoints)

print("Manhattan Average: ", sum(manhatRewSuc) / len(manhatRewSuc))
print("Random Average: ", sum(randRewSuc) / len(randRewSuc))
print("Hill Climbing Average: ", sum(hillClimbRewSuc) / len(hillClimbRewSuc))
print("Simulated Annealing Average: ", sum(simAnnealRewSuc) / len(simAnnealRewSuc))
print("Q-Learn Average: ", sum(qLearnRewSuc) / len(qLearnRewSuc))

# Show the plots
pyplot.show()

# endregion
