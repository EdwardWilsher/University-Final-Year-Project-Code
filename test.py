import copt
import math
import bruteForce
import heuristics
import rlAgents

problem1 = [(439, 520, 1464, 511), (532, 549, 1519, 476), (536, 482, 1529, 527), (495, 452, 1489, 488)]

problem2 = [(532, 549, 1519, 476), (439, 520, 1464, 511), (495, 452, 1489, 488), (536, 482, 1529, 527)]

qLearnAgent = rlAgents.QLearningAgent(4, 1)

order1, success1, result1 = qLearnAgent.getSolution(problem1)
order2, success2, result2 = qLearnAgent.getSolution(problem2)

print(order1, ":", order2)
print(success1, ":", success2)
print(result1, ":", result2)

problem = [(452, 524, 1548, 488), (481, 566, 1461, 474), (568, 411, 1493, 538)]
order = [0, 2, 1]
result = copt.evaluate(problem, order)
print(result)
