import numpy as np
import random
import torch
import math
POOLSIZE = 10
POOLSIZEMAX = 20
FOREIGN = 2
MUTATION = 0.5
NOISE = 0.5

class Genetic:
    def __init__(self, length, popsize):
        self.parents = []
        self.popsize = popsize
        self.length = length
        # np.random.seed(1)
    
    def init(self):
        for i in range(self.popsize):
            self.parents.append(np.random.choice(range(1,2), self.length, replace=True).tolist())
        return self.parents

    def next(self, selectedProbs):
        if len(selectedProbs) == 0:
            res = self.init()
            return res
        pool = []
        otherProbs = []
        print('past:', self.parents)
        print('selected:', selectedProbs)
        for parent in self.parents:
            if parent not in selectedProbs:
                otherProbs.append(parent)
                if np.random.choice([True, False],p=[0.2,0.8]):
                    pool.append(parent)
        
        # print('pool:',pool)
        # print('otherProbs:',otherProbs)
        if len(otherProbs) != 0:
            for i in range(len(otherProbs)):
                resParent = self.crossover(otherProbs[i], selectedProbs[np.random.choice(len(selectedProbs))])
                pool.append(resParent)
        while len(pool) < POOLSIZE:
            pool.append(selectedProbs[np.random.choice(len(selectedProbs))])
        while len(pool) > POOLSIZEMAX:
            pool.pop(0)
        nextParents = self.mutate(pool)
        # print('pool:',pool)
        print('next:', nextParents)
        self.parents = nextParents
        return nextParents

    def crossover(self, parent1, parent2):
        child = []
        
        geneA = int(random.random() * len(parent1))
        geneB = int(random.random() * len(parent1))

        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        for i in range(len(parent1)):
            if i < startGene or i > endGene:
                child.append(parent2[i])
            else:
                child.append(parent1[i])

        return child

    def mutate(self, pool):
        numToMutate = np.random.choice(len(pool)+1)
        parentsToMutate = []
        res = []
        # print('mutatepool:',pool)
        for i in range(numToMutate):
            parent = pool[np.random.choice(len(pool))]
            if parent not in parentsToMutate:
                parentsToMutate.append(parent)
        # print('tomutate:', parentsToMutate)
        for parent in pool:
            if parent not in parentsToMutate:
                res.append(parent)
            else:
                lengthOfParent = len(parent)
                bitToMutate = np.random.choice(lengthOfParent)
                if np.random.choice([True, False]):
                    parent[bitToMutate] += 0.5
                else:
                    parent[bitToMutate] -= 0.5
                    if parent[bitToMutate] <= 0:
                        parent[bitToMutate] = 0.5
                res.append(parent)
        return res

    # def next(self, latent_variables, indices):
    #     variables = c.toTensor(latent_variables)
    #     selected = torch.LongTensor(indices)
    #     pop_size, latent_size = variables.size()
    #     select_size = selected.size(0)
    #     crossover_size = int(max(pop_size - select_size - FOREIGN, 0))
    #     foreign_size = int(min(pop_size - select_size, FOREIGN))

    #     #Crossover
    #     A = variables[c.randomChoice(selected, crossover_size)]
    #     B = variables[c.randomChoice(selected, crossover_size)]
    #     T = torch.Tensor(crossover_size).uniform_()
    #     crossover = _slerp(T, A, B)

    #     #New random members
    #     new = torch.Tensor(foreign_size, latent_size).normal_()

    #     #Mutate
    #     next_pop = torch.cat([variables[selected], crossover])
    #     next_pop = _mutate(next_pop, NOISE, MUTATION)
    #     next_pop = torch.cat([next_pop, new])

    #     return next_pop

    # def _slerp(t, a, b):
    #     omega = torch.sum(torch.renorm(a, 2, 0, 1)*torch.renorm(b, 2, 0, 1), 1)
    #     omega = torch.acos(torch.min(torch.max(-torch.ones(1), omega), torch.ones(1))) + 1e-8

    #     weighted_a = torch.t(torch.sin(omega*(torch.ones(1) - t))/torch.sin(omega)*torch.t(a))
    #     weighted_b = torch.t(torch.sin(omega*t)/torch.sin(omega)*torch.t(b))
    #     return weighted_a + weighted_b

    # def _mutate(a, prob):
    #     size = a.size(0)
    #     mask = torch.Tensor(size).bernoulli_(prob)
    #     t = -math.log(1-.9*NOISE, 10)*mask
    #     b = torch.zeros_like(a).uniform_()
    #     return _slerp(t, a, b)

    # def randomChoice(tensor, n):
    #     size = tensor.size(0)
    #     mask = torch.multinomial(torch.ones(size), n, replacement=True)
    #     return tensor[mask]

if __name__=='__main__':
    # parent1 = [1,2,3,4,5]
    # parent2 = [6,7,8,9,2]
    # res = breed(parent1, parent2)
    # print(res)
    test = Genetic(4, 5)
    res = test.init()
    print('parent:',res)
    nextGen = test.next([[4, 5, 1, 2], [4, 1, 1, 2]])
    print('next:', nextGen)
    # pool = [[1,2,3,1],[1,2,3,5],[2,3,5,1],[5,1,2,5]]
    # numToMutate = np.random.choice(range(1, len(pool)))
    # parentsToMutate = []
    # for i in range(numToMutate):
    #     parent = pool[np.random.choice(len(pool))]
    #     parentsToMutate.append(parent)
    # # parentToMutate = np.random.choices(pool, numToMutate)
    # print(parentsToMutate)
    # nexta = np.random.choice(5,6)
    # print(nexta)
    # print(res)
    # print(res[nexta])
    # a = np.random.choice(5, 5, replace=True)
    # print(a)
    # 
    # a = [[1,2,3,4],[55,5,3,2],[1,2,3,5],[1,2,3,4]]
    # pool = []
    # for i in range(10):
    #     pool.append(a[np.random.choice(len(a))])
    # print(pool)


