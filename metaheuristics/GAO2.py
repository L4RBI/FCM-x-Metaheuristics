from tools import *

import numpy
import math

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

creator.create("Fitness", base.Fitness, weights=(-1.0,)) #creating an object Fitness with deap.
creator.create("Agent", list, fitness = creator.Fitness, velocity = list) #creating an object Agent with deap.


#___________________________________________________________________________

path = "..\Images\Images\T07.JPG"
histogram = Histogram(path)
u = numpy.amax(histogram)
l = 0
ubound = 256
lbound = 0
size = 3
m = 2    
#____________________________________________________________________________
#
def S(r, f, l):
    return (f * numpy.exp(-r / l)) - numpy.exp(-r)


#calculates the euclidean distanc ebetween two points [].
def dis(a, b):
    d = map(operator.add, a, b)  # (subtracting element by element) ^ 2
    d = (_ ** 2 for _ in d)
    sum = 0
    for i in d:
        sum = sum + i
    return numpy.sqrt(sum)

  
#calculates the C based on the 2.8 equation of the paper.
def compute_c(g, max_iter):
    cmax = 1
    cmin = 0.00000001
    return cmax - g * ((cmax - cmin) / max_iter)  #equation 2.8


#calculates the "{ }" part of the 2.7 equation of the paper.
def compute_braquet(agent1 , agent2, c, f, l):#the braquet of the 2.7 equation
    distance = dis(agent1, agent2)
    #right = (agent1 - agent2) / distance 
    right = map(operator.sub, agent2, agent1)
    right = [_ / distance for _ in right ]
    function = 2 + (distance % 2) #norm the distance
    s_thing = S(function, f = f, l = l)
    temp = ((ubound - lbound) * c / 2) * s_thing
    return list((_ * temp for _ in right ))




def generate():
    agent = creator.Agent([random.uniform(ubound, lbound), random.uniform(u, 0)] for _ in range(size))
    agent.best = agent
    agent.fitness = toolbox.evaluate(agent = agent)
    return agent

def updateGrassHopper(agent , Population, c, f , l , best):
    sigma = numpy.zeros((size, 2))
    for p in Population:
        if not numpy.array_equal(p, Population):
            for i in range(size):
                sigma[i] += compute_braquet(agent1 = agent[i], agent2 = p[i], c = c, f = f, l = l) # the sum of the "{ }" part of equation 2.7.
    b = numpy.multiply(best,[[random.random(),random.random()] for _ in range(size)]) #randomizing the second term
    sigma = list(c * sigma * [[random.random(),random.random()] for _ in range(size)] + b  ) #the result with radomization of both terms.
    
    agent[:] = sigma #updating the postion.
    agent.fitness = toolbox.evaluate(agent = agent)
  
def Evaluate(data, agent):
    M = membership(data, centers = agent, m = m)
    return (J(data, M, agent, m = m),)

#setting up the functions for easier calls using the toolbox provided by deap
toolbox = base.Toolbox()
toolbox.register("agent", generate) #setting agent as the function that initialize the agent with the function generate and default args.
toolbox.register("swarm", tools.initRepeat, list, toolbox.agent) #intrepeat helps with repeating the call of the function n times.
toolbox.register("update", updateGrassHopper, f = 0.5, l = 1.5) #registering the updateGrassHopper function as update with setting the default values for some of the args.
toolbox.register("evaluate", Evaluate, data = histogram) #the function used to calculate the fitness set and evaluate.

def main():
    Swarm = toolbox.swarm(n = 40) #intializing the swarm with n agents.
    GEN = 1400 #the numer of max iterations.
    best = None #initializing the best as none.
    #print(Swarm)


    for agent in Swarm: #finding the best position for the initial swarm/population
        if not best or best.fitness > agent.fitness:
            best = creator.Agent(agent)
            best.fitness = agent.fitness

    #print(best)

    for g in range(GEN): #runing the GAO Gen times
        print(Swarm)
        c = compute_c(g + 1, GEN) #computing c 
        for agent in Swarm: #updating each agent using the best solution found 
            toolbox.update(agent = agent, Population = Swarm, c = c, best = best )
        for agent in Swarm: #updating the best solution after updating the whole swarm.
            if not best or best.fitness > agent.fitness:
                best = creator.Agent(agent)
                best.fitness = agent.fitness
        print(best,best.fitness)
        

if __name__ == "__main__":
    main()