import numpy
import copt
import matplotlib.pyplot as pyplot
import heuristics
import rlAgents
import time
import utilities
import threading

# region User Input

# Allow the user to input the number of problems the agent will test their learning with
numberToTest = int(input("Please input the number of problems the RL Agent will test their learning with: "))

# Number of points in each problem
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

# Get the mid time
midTime = time.time()

# Allow the agent to learn
qLearningAgent.learn(numberToTest, squareSize)

# Gets the time after the testing has finished and calculates how long the testing took
timeTaken = time.time() - midTime
print("Testing took", timeTaken, "seconds")

# endregion

# region Output Successes

# Gets the end time and calculates how long the program took
timeTaken = time.time() - startTime
print("The total time that the program took, in seconds, was", timeTaken)

# Gets the successes and rewards
successes, rewards = qLearningAgent.getLearnValues()

# Just get the values that we are testing with
successes = successes[numberToLearn:]
rewards = rewards[numberToLearn:]

# Print the number of successes
print("Number of Successes: ", successes.count(1))

# Get the x-axis
xAxis = [1, 2, 3, 4]

# Splits the list
firstValue = (successes[:int(numberToTest / 4)].count(1) * 100.0) / (numberToTest / 4)
secondValue = (successes[int(numberToTest / 4):int(numberToTest / 2)].count(1) * 100.0) / (numberToTest / 4)
thirdValue = (successes[int(numberToTest / 2):int((3 * numberToTest) / 4)].count(1) * 100.0) / (numberToTest / 4)
fourthValue = (successes[int((3 * numberToTest) / 4):].count(1) * 100.0) / (numberToTest / 4)

# Plots the successes graph
pyplot.bar(xAxis, [firstValue, secondValue, thirdValue, fourthValue], tick_label = ["<25%", ">25% & <50%", ">50% & <75%", ">75%"], width = 0.8)

# Name the x-axis, y-axis and Title
pyplot.xlabel("Problem Percentage")
pyplot.ylabel("Success Percentage")
pyplot.title("Successes at Different Points Throughout Learning")

# Print the results
print("<25%: ", firstValue)
print(">25% & <50%: ", secondValue)
print(">50% & <75%: ", thirdValue)
print(">75%: ", fourthValue)

# Show the plot
pyplot.show()

# endregion

# region Output Rewards

# Get the x-axis
xAxis = list(range(1, numberToTest+1))

# Edit the rewards so that it is smooth
validReward = False
for index in range(0, len(rewards)):
    if ((rewards[index] != 0) and (validReward is False)):
        # If all previous rewards were 0 but this one is valid then set all previous rewards to this value
        validReward = True
        for index2 in range(0, index):
            rewards[index2] = rewards[index]
    elif ((rewards[index] == 0) and (validReward is True)):
        # Set this reward to the previous reward
        rewards[index] = rewards[index - 1]

# Print the average reward
print("Average Reward: ", (sum(rewards) / float(len(rewards))))

# Plots the rewards graph
pyplot.plot(xAxis, rewards)

# Name the x-axis, y-axis and Title
pyplot.xlabel("Problem Number")
pyplot.ylabel("Reward")
pyplot.title("RL Agent Rewards")

# Show the plot
pyplot.show()

# endregion
