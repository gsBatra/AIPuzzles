########################################################
#
# Genetic Algorithm
#
########################################################



student_name = 'Gagandeep Batra'
student_email = 'gsb5195@psu.edu'



########################################################
# Import
########################################################

from utils import *
import math
import random


################################################################
# 1. Genetic Algorithm
################################################################


def genetic_algorithm(problem, f_thres, ngen=1000):
    """
    Returns a tuple (i, sol) 
    where
      - i  : number of generations computed
      - sol: best chromosome found
    """
    population = problem.init_population()
    best = problem.fittest(population, f_thres)
    if best in population:
        return -1, best
    for i in range(ngen):
        population = problem.next_generation(population)
        best = problem.fittest(population, f_thres)
        if best in population:
            return i, best
    best = problem.fittest(population)
    return ngen, best

  
################################################################
# 2. NQueens Problem
################################################################


class NQueensProblem(GeneticProblem):
    def __init__(self, n, g_bases, g_len, m_prob):
        super().__init__(n, g_bases, g_len, m_prob)
    
    def init_population(self):
        population = []
        for x in range(self.n):
            chromosome = []
            for y in range(self.g_len):
                chromosome.append(random.choice(self.g_bases))
            population.append(tuple(chromosome))
        return population
 
    def next_generation(self, population):
        next_gen = []
        while len(next_gen) != len(population):
            x = random.choice(population)
            y = random.choice(population)
            child = self.crossover(x, y)
            child = self.mutate(child)
            if child not in population:
                next_gen.append(child)
        return next_gen
        
    def mutate(self, chrom):
        p = random.uniform(0,1)
        if p > self.m_prob:
            return chrom
        i = random.randrange(self.g_len)
        chrom = list(chrom)
        chrom[i] = random.choice(self.g_bases)
        return tuple(chrom)
    
    def crossover(self, chrom1, chrom2):
        i = random.randrange(self.g_len)
        return tuple(list(chrom1[:i]) + list(chrom2[i:]))
    
    def fitness_fn(self, chrom):
        n = len(chrom)
        max_fitness = math.comb(n, 2)
        chrom = list(chrom)
        horizontal_collisions = sum([chrom.count(queen)-1 for queen in chrom])/2

        locations = [(r,c) for c,r in enumerate(chrom)]
        arr = [[0 for c in range(n)] for r in range(n)]

        for r in range(n):
            for c in range(n):
                if locations[c][0] == r and locations[c][1] == c:
                    arr[r][c] = 1
        diagonal_collisions = 0
        for x in locations:
            diagonal_collisions += self.dconflicts(arr, x)

        return int(max_fitness - (horizontal_collisions + diagonal_collisions))

    def dconflicts(self, board, location):
        count = 0
        row = location[0]
        col = location[1]
        
        for r, c in zip(range(row-1, -1, -1), range(col+1, self.g_len, 1)): 
            if board[r][c] == 1:
                count+=1
        for r, c in zip(range(row+1, self.g_len, 1), range(col+1, self.g_len, 1)):
            if board[r][c] == 1:
                count+=1
        return count
    
    def select(self, m, population):
        if m == 0:
            return []
        total_fitness = sum([self.fitness_fn(x) for x in population])     
        prob_dist = [self.fitness_fn(x)/total_fitness for x in population]
        cum_dist = []
        cumsum = 0
        for x in prob_dist:
            cumsum += x
            cum_dist.append(cumsum)
            
        chromosomes = []
        for x in range(m):
            p = random.uniform(0, 1)
            for i, prob in enumerate(cum_dist):
                if p >= prob:
                    continue
                chromosomes.append(population[i])
                break
        return chromosomes
            
    def fittest(self, population, f_thres=None):
        fitness_values = list(map(self.fitness_fn, population))
        best_chrom = population[fitness_values.index(max(fitness_values))]
        if f_thres == None:
            return best_chrom
        if max(fitness_values) >= f_thres:
            return best_chrom
        return None
        
################################################################
# 3. Function Optimaization f(x,y) = x sin(4x) + 1.1 y sin(2y)
################################################################


class FunctionProblem(GeneticProblem):
    def __init__(self, n, g_bases, g_len, m_prob):
        super().__init__(n, g_bases, g_len, m_prob)

    def init_population(self):
        return [(random.uniform(0, self.g_bases[0]), random.uniform(0, self.g_bases[1])) for x in range(self.n)] 

    def next_generation(self, population):
        fitness_list = list(map(self.fitness_fn, population))
        fitness_list.sort()
        ranked_chromosomes = [y for x in fitness_list for y in population if self.fitness_fn(y) == x]
        best_half = ranked_chromosomes[:int(len(population)/2)]
        next_gen = []
        while len(next_gen) != (len(population)-len(best_half)):
            x = random.choice(best_half)
            y = random.choice(best_half)
            child = self.crossover(x, y)
            child = self.mutate(child)
            if child not in best_half:
                next_gen.append(child)
        best_half.extend(next_gen)
        return best_half
    
    def mutate(self, chrom):
        p = random.uniform(0,1)
        if p > self.m_prob:
            return chrom
        i = random.randrange(len(chrom))
        chrom = list(chrom)
        chrom[i] = random.uniform(0, self.g_bases[i])
        return tuple(chrom)
        
    def crossover(self, chrom1, chrom2):
        x1,y1,x2,y2 = chrom1[0],chrom1[1],chrom2[0],chrom2[1]
        component = random.choice(('x', 'y'))
        alpha = random.uniform(0, 1)
        if component == 'x':
            return ((1-alpha) * x1 + alpha * x2, y1)
        return (x1, (1-alpha) * y1 + alpha * y2)
    
    def fitness_fn(self, chrom):
        x = chrom[0]
        y = chrom[1]
        return x * math.sin(4*x) + 1.1 * y * math.sin(2*y)
    
    def select(self, m, population):
        if m == 0:
            return []
        n = len(population)
        fitness_list = list(map(self.fitness_fn, population))
        fitness_list.sort()
        ranked_chromosomes = [y for x in fitness_list for y in population if self.fitness_fn(y) == x]
        total = sum([x for x in range(n+1)])
        prob_dist = [(n-k)/total for k in range(n)]
        cum_dist = []
        cumsum = 0
        for x in prob_dist:
            cumsum += x
            cum_dist.append(cumsum)
        chromosomes = []
        for x in range(m):
            p = random.uniform(0, 1)
            for i, prob in enumerate(cum_dist):
                if p >= prob:
                    continue
                chromosomes.append(ranked_chromosomes[i])
                break
        return chromosomes

    def fittest(self, population, f_thres=None):
        fitness_values = list(map(self.fitness_fn, population))
        best_chrom = population[fitness_values.index(min(fitness_values))]
        if f_thres == None:
            return best_chrom
        if min(fitness_values) <= f_thres:
            return best_chrom
        return None


################################################################
# 4. Traveling Salesman Problem
################################################################


class HamiltonProblem(GeneticProblem):
    def __init__(self, n, g_bases, g_len, m_prob, graph=None):
        super().__init__(n, g_bases, g_len, m_prob)
        self.graph = graph

    def init_population(self):
        pop = []
        for x in range(self.n):
            random.shuffle(self.g_bases)
            pop.append(tuple(self.g_bases))
        return pop
    
    def next_generation(self, population):
        while len(population) != 2 * self.n:
            x = random.choice(population)
            y = random.choice(population)
            child = self.crossover(x, y)
            child = self.mutate(child)
            if child not in population:
                population.append(child)
        fitness_list = list(map(self.fitness_fn, population))
        fitness_list.sort()
        new_gen = [y for x in fitness_list for y in population if self.fitness_fn(y) == x]
        return new_gen[:self.n]        
          
    def mutate(self, chrom):
        p = random.uniform(0,1)
        if p > self.m_prob:
            return chrom
        index1 = random.randrange(len(chrom))
        index2 = random.randrange(len(chrom))
        chrom = list(chrom)
        chrom[index1], chrom[index2] = chrom[index2], chrom[index1]
        return tuple(chrom)
    
    def crossover(self, chrom1, chrom2):
        index = random.randrange(len(chrom1))
        chrom1, chrom2 = list(chrom1), list(chrom2)
        chrom1[index], chrom2[index] = chrom2[index], chrom1[index]
        while len(chrom1) != len(set(chrom1)):
            duplicate_value = ''
            duplicate_indexes = [i for i, x in enumerate(chrom1) if chrom1.count(x) > 1]
            for x in duplicate_indexes:
                if x != index:
                    index = x
                    chrom1[x], chrom2[x] = chrom2[x], chrom1[x]
                    break
        return tuple(chrom1)
            
    def fitness_fn(self, chrom):
        chrom = list(chrom)
        n = len(chrom)
        total = 0
        for x in range(n-1):
            start = self.graph.get(chrom[x])
            total += start.get(chrom[x+1])
        cn = self.graph.get(chrom[n-1])
        return total + cn.get(chrom[0])

    def select(self, m, population):
        if m == 0:
            return []
        fitness_list = list(map(self.fitness_fn, population))
        T = sum(fitness_list)
        S = sum([(T-x) for x in fitness_list])
        prob_dist = [(T - x)/S for x in fitness_list]
        cum_dist = []
        cumsum = 0
        for x in prob_dist:
            cumsum += x
            cum_dist.append(cumsum)
        chromosomes = []
        for x in range(m):
            p = random.uniform(0, 1)
            for i, prob in enumerate(cum_dist):
                if p >= prob:
                    continue
                chromosomes.append(population[i])
                break
        return chromosomes
        
    def fittest(self, population, f_thres=None):
        fitness_values = list(map(self.fitness_fn, population))
        best_chrom = population[fitness_values.index(min(fitness_values))]
        if f_thres == None:
            return best_chrom
        if min(fitness_values) <= f_thres:
            return best_chrom
        return None

