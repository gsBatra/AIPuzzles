student_name = 'Gagandeep Batra'
student_email = 'gsb5195@psu.edu'

########################################################
# Import
########################################################

from utils import *
from collections import deque
import math
from queue import PriorityQueue

##########################################################
# 1. Best-First, Uniform-Cost, A-Star Search Algorithms
##########################################################

def best_first_search(problem):
    node = Node(problem.init_state, heuristic=problem.h(problem.init_state))
    explored = {node.state: node}          

    frontier = PriorityQueue()
    frontier.put((node.heuristic, 0, node))
    count=1

    while frontier:
        node = frontier.get()[2]
        if problem.goal_test(node.state):
            return node
        for child in node.expand(problem):
            if child.state not in explored or child.heuristic < explored[child.state].heuristic:
                explored[child.state] = child
                frontier.put((child.heuristic, count, child))
                count+=1
    return None

def uniform_cost_search(problem):
    node = Node(problem.init_state)
    explored = {node.state: node}  

    frontier = PriorityQueue()
    frontier.put((node.path_cost, 0, node))
    count=1
    
    while frontier:
        node = frontier.get()[2]
        if problem.goal_test(node.state):
            return node
        for child in node.expand(problem):
            if child.state not in explored or child.path_cost < explored[child.state].path_cost:
                explored[child.state] = child
                frontier.put((child.path_cost, count, child))
                count+=1
    return None
    
def a_star_search(problem):
    node = Node(problem.init_state, heuristic=problem.h(problem.init_state))
    explored = {node.state: node}  

    frontier = PriorityQueue()
    frontier.put((node.heuristic + node.path_cost, 0, node))
    count=1

    while frontier:
        node = frontier.get()[2]
        if problem.goal_test(node.state):
            return node
        for child in node.expand(problem): 
            if child.state not in explored or child.heuristic + child.path_cost < explored[child.state].heuristic + explored[child.state].path_cost:
                explored[child.state] = child
                frontier.put((child.heuristic + child.path_cost, count, child))
                count+=1
    return None

##########################################################
# 2. N-Queens Problem
##########################################################


class NQueensProblem(Problem):
    """
    The implementation of the class NQueensProblem is given
    for those students who were not able to complete it in
    Homework 2.
    
    Note that you do not have to use this implementation.
    Instead, you can use your own implementation from
    Homework 2.

    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    """
    
    def __init__(self, n):
        super().__init__(tuple([-1] * n))
        self.n = n
        
    def actions(self, state):
        if state[-1] != -1:   # if all columns are filled
            return []         # then no valid actions exist
        
        valid_actions = list(range(self.n))
        col = state.index(-1) # index of leftmost unfilled column
        for row in range(self.n):
            for c, r in enumerate(state[:col]):
                if self.conflict(row, col, r, c) and row in valid_actions:
                    valid_actions.remove(row)
                    
        return valid_actions
    
    def result(self, state, action):
        col = state.index(-1) # leftmost empty column
        new = list(state[:])  
        new[col] = action     # queen's location on that column
        return tuple(new)

    def goal_test(self, state):
        if state[-1] == -1:   # if there is an empty column
            return False;     # then, state is not a goal state

        for c1, r1 in enumerate(state):
            for c2, r2 in enumerate(state):
                if (r1, c1) != (r2, c2) and self.conflict(r1, c1, r2, c2):
                    return False
        return True

    def conflict(self, row1, col1, row2, col2):
        return row1 == row2 or col1 == col2 or abs(row1-row2) == abs(col1-col2)
        
    def g(self, cost, from_state, action, to_state):
        """
        Return path cost from start state to to_state via from_state.
        The path from start_state to from_state costs the given cost
        and the action that leads from from_state to to_state
        costs 1.
        """
        return self.n - to_state.count(-1)


    def h(self, state):
        """
        Returns the heuristic value for the given state.
        Use the total number of conflicts in the given
        state as a heuristic value for the state.
        """
        locations = [ (r,c) for c,r in enumerate(state)]
        arr = [[0 for c in range(self.n)] for r in range(self.n)]
        negRow = [[ 1 if locations[r][0] == -1 else 0 for r in range(self.n)]]
        for r in range(self.n):
            for c in range(self.n):
                if locations[c][0] != -1 and locations[c][0] == r and locations[c][1] == c:
                    arr[r][c] = 1
        negRow.extend(arr)
        count = 0
        for x in range(self.n):
            count += self.hconflict(negRow, locations[x])
        return count

    def hconflict(self, board, location):
        count=0
        row = location[0]+1
        col = location[1]
        
        for c in range(col-1, -1, -1): #check each column
            if board[row][c] == 1:
                count+=1
        for c in range(col+1, self.n, 1):
            if board[row][c] == 1:
                count+=1
                
        for r in range(row-1, -1, -1): #check each row
            if board[r][col] == 1:
                count+=1
        for r in range(row+1, self.n+1, 1):
            if board[r][col] == 1:
                count+=1
                
        for r, c in zip(range(row-1, -1, -1), range(col-1, -1, -1)): #check left diagonals
            if board[r][c] == 1:
                count+=1
        for r, c in zip(range(row+1, self.n+1, 1), range(col-1, -1, -1)):
            if board[r][c] == 1:
                count+=1

        for r, c in zip(range(row-1, -1, -1), range(col+1, self.n, 1)): #check right diagonals
            if board[r][c] == 1:
                count+=1
        for r, c in zip(range(row+1, self.n+1, 1), range(col+1, self.n, 1)):
            if board[r][c] == 1:
                count+=1
                
        return count
    
##########################################################
# 3. Graph Problem
##########################################################



class GraphProblem(Problem):
    """
    The implementation of the class GraphProblem is given
    for those students who were not able to complete it in
    Homework 2.
    
    Note that you do not have to use this implementation.
    Instead, you can use your own implementation from
    Homework 2.

    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    """
    
    
    def __init__(self, init_state, goal_state, graph):
        super().__init__(init_state, goal_state)
        self.graph = graph

        
    def actions(self, state):
        """Returns the list of adjacent states from the given state."""
        return list(self.graph.get(state).keys())

    
    def result(self, state, action):
        """Returns the resulting state by taking the given action.
            (action is the adjacent state to move to from the given state)"""
        return action

    
    def goal_test(self, state):
        return state == self.goal_state

    
    def g(self, cost, from_state, action, to_state):
        """
        Returns the path cost from root to to_state.
        Note that the path cost from the root to from_state
        is the give cost and the given action taken at from_state
        will lead you to to_state with the cost associated with
        the action.
        """
        for k,v in self.graph.get(from_state).items():
            if to_state == k:
                return cost + v
        return 0

    def h(self, state):
        """
        Returns the heuristic value for the given state. Heuristic
        value of the state is calculated as follows:
        1. if an attribute called 'heuristics' exists:
           - heuristics must be a dictionary of states as keys
             and corresponding heuristic values as values
           - so, return the heuristic value for the given state
        2. else if an attribute called 'locations' exists:
           - locations must be a dictionary of states as keys
             and corresponding GPS coordinates (x, y) as values
           - so, calculate and return the straight-line distance
             (or Euclidean norm) from the given state to the goal
             state
        3. else
           - cannot find nor calculate heuristic value for given state
           - so, just return a large value (i.e., infinity)
        """

        if hasattr(self.graph, 'heuristics'):
            for k in self.graph.heuristics.keys():
                if state == k:
                    return self.graph.heuristics.get(k)
        elif hasattr(self.graph, 'locations'):
            for k in self.graph.locations.keys():
                if state == k:
                    start = self.graph.locations.get(k)
                    goal = self.graph.locations.get(self.goal_state)
                    return math.dist(start, goal)
        else:
            return math.inf



##########################################################
# 4. Eight Puzzle
##########################################################


class EightPuzzle(Problem):
    def __init__(self, init_state, goal_state=(1,2,3,4,5,6,7,8,0)):
        super().__init__(init_state, goal_state)
    

    def actions(self, state):
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        blank_tile = state.index(0)
        
        if blank_tile % 3 == 0:     #first column
            actions.remove('LEFT')
        if blank_tile < 3:          #first row
            actions.remove('UP')
        if blank_tile % 3 == 2:     #third column
            actions.remove('RIGHT')
        if blank_tile > 5:          #third row
            actions.remove('DOWN')

        return actions

    
    def result(self, state, action):
        blank_tile = state.index(0)
        result = list(state)
        
        if action == 'UP':  #move up 3 elements
            result[blank_tile], result[blank_tile-3] = result[blank_tile-3], result[blank_tile]
            return tuple(result)
        elif action == 'DOWN': #move down 3 elements
            result[blank_tile], result[blank_tile+3] = result[blank_tile+3], result[blank_tile]
            return tuple(result)
        elif action == 'LEFT': #move left 1 element
            result[blank_tile], result[blank_tile-1] = result[blank_tile-1], result[blank_tile]
            return tuple(result)
        else:   #move right 1 element
            result[blank_tile], result[blank_tile+1] = result[blank_tile+1], result[blank_tile]
            return tuple(result)

    def goal_test(self, state):
        return state == self.goal_state

    
    def g(self, cost, from_state, action, to_state):
        """
        Return path cost from root to to_state via from_state.
        The path from root to from_state costs the given cost
        and the action that leads from from_state to to_state
        costs 1.
        """
        if action == '' or action == None:
            return cost
        else:
            return cost+1

    
    def h(self, state):
        """
        Returns the heuristic value for the given state.
        Use the sum of the Manhattan distances of misplaced
        tiles to their final positions.
        """
        if state == None:
            return 0
        else:
            distance = 0
            for x,value in enumerate(state):
                if value != 0:
                    state_row,goal_row = int(x/3),int((value-1)/3)
                    state_col,goal_col = x%3,(value-1)%3
                    distance += abs(state_row-goal_row) + abs(state_col-goal_col)
            return distance

