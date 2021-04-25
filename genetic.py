#!/usr/bin/python3.8

import time
import copy
import random
import numpy as np
import clustering as clt

class HyperParams:
    def __init__(self, population_size : int, crossover_rate : float, mutation_rate : float):
        self._population_size = population_size
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate
    
    @property
    def population_size(self) -> int:
        return self._population_size
    
    @property
    def crossover_rate(self) -> float:
        return self._crossover_rate
    
    @property
    def mutation_rate(self) -> float:
        return self._mutation_rate

class Population:
    def __init__(self, k : int, size : int, clusters : clt.Clusters):
        self._size = size
        self._instances : list = []
        self._k : int = k
        self._point_dim : tuple = clusters.point_dim
        for i in range(size):
            self._instances.append(clt.Clusters(clusters.k, clusters.points))
            self._instances[i].initialize_state()
    
    @property
    def instances(self) -> list:
        return self._instances

    def mutate(self, instance : int):
        self._instances[instance].disturb()
        self._instances[instance].accept_disturbed()
    
    def crossover(self, a : int, b : int) -> clt.Clusters:
        offspring : clt.Clusters = clt.Clusters(self._instances[a].k, self._instances[a].points)
        
        divide : int = self._k // 2 if self._k <= 3 else random.randint(1, self._k - 2)
        centroids : list = list([])
        for i in range(self._k):
            centroids.append(self._instances[a].centroids[i] if i < divide else self._instances[b].centroids[i])
        
        offspring.initialize_state(centroids)

        return offspring
    
    def adaptation(self, instance : int) -> float:
        sse_sum : float = 0
        for i in range(self._size):
            sse_sum += self._instances[i].sse
        return self._instances[i].sse / sse_sum
    
    def instance_value(self, instance : int) -> float:
        return self._instances[instance].sse
    
    def coverging(self) -> bool:
        s : float = self._instances[0].sse
        for i in range(1, self._size):
            if abs(s - self._instances[i].sse) > .000000001:
                return False
        return True
    
    def set_instance(self, instance : int, clusters : clt.Clusters):
        self._instances[instance] = clusters

    def set_instances(self, instances : list):
        for i in range(self._size):
            self._instances[i] = instances[i]

def genetic(hyper_params : HyperParams, clusters : clt.Clusters, population : Population = None) -> (float, clt.Clusters):

    population = Population(clusters.k, hyper_params.population_size, clusters) if population == None else population
    
    while not population.coverging():

        population_indices : list = range(0, hyper_params.population_size)
        probabilities : list = list(map(population.adaptation, population_indices))
        values : list = list(map(population.instance_value, population_indices))
        a : int = random.choices(population_indices, probabilities, k = 1)[0]
        b : int = random.choices(population_indices, probabilities, k = 1)[0]
        if random.random() < hyper_params.crossover_rate:
            offspring : clt.Clusters = population.crossover(a, b)
            max_value : float = max(values)
            if offspring.sse < max_value:
                population.set_instance(values.index(max_value), offspring)
    
    result : float = min(map(population.instance_value, range(0, hyper_params.population_size)))
    result_instace : clt.Clusters = None

    for instance in population.instances:
        if instance.sse == result:
            result_instace = instance

    return (result, result_instace)