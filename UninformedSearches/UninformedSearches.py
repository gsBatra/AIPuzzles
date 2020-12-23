name = 'Gagandeep Batra'
email = 'gsb5195@psu.edu'




########################################################
# Import
########################################################


from utils import *
from collections import deque

# Add your imports here if used





##########################################################
# 1. Uninformed Any-Path Search Algorithms
##########################################################


def depth_first_search(problem):
    
    node = Node(problem.init_state)
    frontier = deque([node])         # stack: append/pop
    explored = [problem.init_state]  # used as "visited"
    
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.append(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
    return Node(None)
    
def breadth_first_search(problem):
    
    node = Node(problem.init_state)
    if problem.goal_test(node.state):
        return node
        
    frontier = deque([node])         # queue: append/popleft
    explored = [problem.init_state]  # used as "visited"
    
    while frontier:
        node = frontier.popleft()
        explored.append(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return Node(None)


##########################################################
# 2. N-Queens Problem
##########################################################


class NQueensProblem(Problem):
    
    def __init__(self, n):
        self.n = n
        if n == None:
            n = 0
        self.initial = tuple([-1] * n)
        Problem.__init__(self, self.initial)

    def actions(self, state):
        if len(state) != self.n:
            return []
        elif len(set(state)) <= 1:
            return list(range(self.n))
        elif -1 not in state:
            return []
        
        board = self.createArray(self.n, state)
        valid = []

        for x in range(self.n):
             if self.isSafe(board, x, len(board[0])):
                 valid.append(x)
        return valid

    def createArray(self, row, state):
        col = state.index(-1)
        leftlist = [ i for i in state [:col] ]
        arr = [[0 for c in range(col)] for r in range(row)]
        for c in range(col):
            for r in range(leftlist[c]+1):
                if r == leftlist[c]:
                    arr[r][c] = 1
        return arr
    
    def isSafe(self, board, row, col):
        for c in range(col):
            if board[row][c] == 1:
                return False
        for r, c in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
            if board[r][c] == 1:
                return False
        for r, c in zip(range(row+1, self.n, 1), range(col-1, -1, -1)):
            if board[r][c] == 1:
                return False
        return True
                
    def result(self, state, action):
        if action not in self.actions(state):
            return []
        elif -1 not in state:
            return state
        else:
            l = list(state)
            for x in range(len(l)):
                if l[x] == -1:
                    l[x] = action
                    return tuple(l)
            
    def goal_test(self, state):
        if self.n == None:
            return True
        if self.n < 1:
            return False
        elif state[-1] == -1:
            return False
        else:
            goal = list(state)
            test = list(state)
            for x in range(1, len(test)):
                test[x] = -1
            while True:
                for x in range(1, len(test)):
                    action = self.actions(test)
                    if goal[x] in action:
                        test[x] = goal[x]
                        continue
                    else:
                        return False
                break
            return True


##########################################################
# 3. Farmer's Problem
##########################################################


class FarmerProblem(Problem):
    
    def __init__(self, init_state, goal_state):
        Problem.__init__(self, init_state, goal_state)
    
    def actions(self, state):
        if len(state) != 4:
            return []
        
        keys = []
        things = ('farmer', 'grain', 'chicken', 'fox')
        combined = dict(zip(things, state))

        if self.isValid(combined):
            if all(value == True for value in combined.values()) or all(value == False for value in combined.values()):
                keys.append('FC')
                return keys
            if combined.get('farmer') == True and combined.get('grain') == True and combined.get('chicken') == True and combined.get('fox') == False or combined.get('farmer') == False and combined.get('grain') == False and combined.get('chicken') == False and combined.get('fox') == True:
                keys.append('FG')
                keys.append('FC')
                return keys
            if combined.get('farmer') == True and combined.get('grain') == True and combined.get('chicken') == False and combined.get('fox') == True or combined.get('farmer') == False and combined.get('grain') == False and combined.get('chicken') == True and combined.get('fox') == False:
                keys.append('F')
                keys.append('FG')
                keys.append('FX')
                return keys
            if combined.get('farmer') == True and combined.get('grain') == False and combined.get('chicken') == True and combined.get('fox') == True or combined.get('farmer') == False and combined.get('grain') == True and combined.get('chicken') == False and combined.get('fox') == False:
                keys.append('FC')
                keys.append('FX')
                return keys
            keys.append('F')
            keys.append('FC')
            return keys
        else:
            return []
                
    def isValid(self, combined):
        chicken_eats_cabbage = (combined.get('chicken') == combined.get('grain') and
                                combined.get('farmer') != combined.get('chicken'))
        fox_eats_chicken = (combined.get('fox') == combined.get('chicken') and
                            combined.get('farmer') != combined.get('fox'))
        
        return not (chicken_eats_cabbage or fox_eats_chicken)

    def result(self, state, action):
        if len(state) != 4:
            return state

        temp = list(state)
        if action == 'F' or action == 'FG' or action == 'FC' or action == 'FX':
            temp[0] = not temp[0]
            if action == 'FG':
                temp[1] = not temp[1]
            elif action == 'FC':
                temp[2] = not temp[2]
            elif action == 'FX':
                temp[3] = not temp[3]
        else:
            return state
        return tuple(temp)
    
    def goal_test(self, state):
        return state == self.goal_state


##########################################################
# 4. Graph Problem
##########################################################


class GraphProblem(Problem):
    
    def __init__(self, init_state, goal_state, graph):
        Problem.__init__(self, init_state, goal_state)
        self.graph = graph

    
    def actions(self, state):
        cities = Graph.get(self.graph, state)
        return list(cities.keys())
                   
    
    def result(self, state, action):
        return action

    
    def goal_test(self, state):
        return state == self.goal_state

