import copt
import math
import bruteForce
import heuristics
import rlAgents
import utilities

problem1 = [(439, 520, 1464, 511), (532, 549, 1519, 476), (536, 482, 1529, 527), (495, 452, 1489, 488)]

problem2 = [(532, 549, 1519, 476), (439, 520, 1464, 511), (495, 452, 1489, 488), (536, 482, 1529, 527)]

qLearnAgent = rlAgents.QLearningAgent(4, 1)

order1, success1, result1 = qLearnAgent.getSolution(problem1)
order2, success2, result2 = qLearnAgent.getSolution(problem2)

print(order1, ":", order2)
print(success1, ":", success2)
print(result1, ":", result2)

problem = [(452, 524, 1548, 488), (481, 566, 1461, 474), (568, 411, 1493, 538)]
order = [1, 0]
result = copt.evaluate(problem, order)
print(result)

# # Permutes the problem so that the point with the smallest x index will be first etc
# # Returns the permuted problem as well as the permutation
# def permuteProblem(problemNew):
#     # Initialises the two arrays that will be returned
#     permutedProblem = [(-1, -1, -1, -1)]*3
#     permutation = [-1]*3
#
#     # Iterates through the problem
#     for j in range(0, 3):
#         lowestVal = (10000, 10000, 10000, 10000)
#         lowestIndex = -1
#         for k in range(0, 3):
#             if (problemNew[k] in permutedProblem):
#                 # Point already in the permuted list so continue
#                 continue
#             else:
#                 # Checks to see if this point is lower than the current lowest
#                 if (problemNew[k][0] < lowestVal[0]):
#                     lowestVal = problemNew[k]
#                     lowestIndex = k
#                 elif (problemNew[k][0] == lowestVal[0]):
#                     if (problemNew[k][1] < lowestVal[1]):
#                         lowestVal = problemNew[k]
#                         lowestIndex = k
#                     elif (problemNew[k][1] == lowestVal[1]):
#                         if (problemNew[k][2] < lowestVal[2]):
#                             lowestVal = problemNew[k]
#                             lowestIndex = k
#                         elif (problemNew[k][2] == lowestVal[2]):
#                             if (problemNew[k][3] < lowestVal[3]):
#                                 lowestVal = problemNew[k]
#                                 lowestIndex = k
#
#         # Adds the lowest point to the permuted list
#         permutedProblem[j] = lowestVal
#         permutation[j] = lowestIndex
#
#     # Returns the new problem and index
#     return permutedProblem, permutation
#
#
# problemsSeen = []
#
# for i in range(0, 100000):
#     problem = utilities.generateSmallerProblem(3, 1)
#     problem, _ = permuteProblem(problem)
#     if not(problem in problemsSeen):
#         problemsSeen.append(problem)
#
#     utilities.outputPercentageComplete(i+1, 100000, len(problemsSeen))
#
# print(len(problemsSeen))

# gamma = 0.1 -> 2.0, 1.9, 1.9, 0.4, 0.7
# gamma = 0.2 -> 1.5, 1.5, 3.6, 0.8, 2.8
# gamma = 0.3 -> 2.1, 1.3, 3.4, 2.1, 4.3
# gamma = 0.4 -> 3.3, 3.4, 2.2, 1.9, 1.9
# gamma = 0.5 -> 2.1, 1.1, 4.3, 4.0, 0.7
# gamma = 0.6 -> 1.8, 3.7, 3.0, 1.8, 1.0
# gamma = 0.7 -> 2.0, 1.9, 2.4, 1.7, 0.4
# gamma = 0.8 -> 0.5, 0.3, 0.1, 0.7, 0.5
# gamma = 0.9 -> 0.8, 0.4, 0.1, 1.8, 0.2
