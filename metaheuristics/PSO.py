import operator
import random

import numpy
import math

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

creator.create("Fitness", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness = creator.Fitness, velocity = list, best = None)


#initialize the Particule randomly
def generate( pmin, pmax , vmin, vmax ):
    particule = creator.Particle([random.uniform(pmin,pmax), random.uniform(pmin, pmax)]) 
    particule.velocity = [random.uniform(vmin, vmax) for _ in range(2)]
    return particule


#Update the postion of a particule using the equations 3 and 4 provided by the paper.
def updateParticle(particule, best, constant1, constant2, weight, vmin, vmax):
    rand1 = (random.uniform(0, 1) for _ in range(len(particule)))
    rand2 = (random.uniform(0, 1) for _ in range(len(particule)))
    rand1 = (_ * constant1 for _ in rand1)
    rand2 = (_ * constant2 for _ in rand2)
    v = (_ * weight for _ in particule.velocity) 
    rand1_local = map(operator.mul, rand1, map(operator.sub, particule.best, particule))
    rand2_global = map(operator.mul, rand2, map(operator.sub, best, particule))
    particule.velocity = list(map(operator.add, v, map(operator.add, rand1_local, rand2_global))) #equation 3

    for i, velocity in enumerate(particule.velocity): #making sure the velocity isn't yoo high.
        if abs(velocity) < vmin:
            particule.velocity[i] = math.copysign(vmin, velocity)
        elif abs(velocity) > vmax:
            particule.velocity[i] = math.copysign(vmax, velocity)
    
    particule[:] = list(map(operator.add, particule, particule.velocity)) #equation 4 

def Evaluate(particule):
    pass

toolbox = base.Toolbox()
toolbox.register("particle", generate, pmin=-100, pmax = 100 ,vmin = -100, vmax = 100)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, constant1 = 2, constant2 =2 , weight = 0.5, vmin = -100 , vmax = 100 )
toolbox.register("evaluate", benchmarks.ackley)

def main():
    Population = toolbox.population(n = 5)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields


    GEN = 1000
    best = None

    for g in range(GEN): #computing the PSO Algorithm GEN times
        for particule in Population:
            particule.fitness.values = toolbox.evaluate(particule)

            if not particule.best or particule.best.fitness < particule.fitness: #updating the personal best.
                particule.best = creator.Particle(particule)
                particule.best.fitness.values = particule.fitness.values

            if not best or best.fitness < particule.fitness: #updating the global best.
                best = creator.Particle(particule)
                best.fitness.values = particule.fitness.values
        
        for particule in Population: #updating the position for each particule.
            toolbox.update(particule, best)

            
         # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(Population), **stats.compile(Population))
        print(logbook.stream)
    #print(best,best.fitness)
    
    return Population, logbook, best

if __name__ == "__main__":
    main()


