import operator 
import random

import numpy
import math

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

creator.create("Fitness", base.Fitness, weights=(-1.0,)) #creating an object Fitness with deap.
creator.create("Agent", list, fitness = creator.Fitness, velocity = list, best = None) #creating an object Agent with deap.

#calculates the S function based on the 2.3 equation of the paper.
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
    cmin = 0.00001
    return cmax - g * ((cmax - cmin) / max_iter)  #equation 2.8


#calculates the "{ }" part of the 2.7 equation of the paper.
def compute_braquet(agent1 , agent2, c, f, l, ubound, lbound):#the braquet of the 2.7 equation
    distance = dis(agent1, agent2)
    #right = (agent1 - agent2) / distance 
    right = map(operator.sub, agent1, agent2)
    right = [_ / distance for _ in right ]
    function = 2 + (distance / 2)
    s_thing = S(function, f = f, l = l)
    temp = ((ubound - lbound) * c / 2) * s_thing
    return list((_ * temp for _ in right ))


#the function that generates the initial an Agent randomly.
def generate(ubound, lbound):
    agent = creator.Agent([random.uniform(ubound, lbound), random.uniform(ubound, lbound)])
    agent.best = agent
    agent.fitness = toolbox.evaluate(agent)
    return agent


#updates the position of the agent using the 2.7 equation of the paper.
def updateGrassHopper(agent , Population, c, f , l , ubound, lbound, best):
    sigma = numpy.zeros(2)
    for p in Population:
        if p != agent:
            sigma += compute_braquet(agent1 = agent, agent2 = p, c = c, f = f, l = l, ubound = ubound, lbound = lbound) # the sum of the "{ }" part of equation 2.7.
    sigma = list(c * sigma + best) #the result.
    """for _ in sigma:
        _[:] = min(_, ubound)
        _[:] = max(_, lbound)"""
    agent[:] = sigma #updating the postion.
    agent.fitness = toolbox.evaluate(agent)
    #keeping track of the personal best position of the agent.
    if agent.best.fitness < agent.fitness:
        print("pbest updated")
        agent.best = creator.Agent(agent)
        agent.best.fitness = agent.fitness


#setting up the functions for easier calls using the toolbox provided by deap
toolbox = base.Toolbox()
toolbox.register("agent", generate, lbound =-15 , ubound =30 ) #setting agent as the function that initialize the agent with the function generate and default args.
toolbox.register("swarm", tools.initRepeat, list, toolbox.agent) #intrepeat helps with repeating the call of the function n times.
toolbox.register("update", updateGrassHopper, f = 1.5, l = 2.5,  lbound = -100, ubound = 100) #registering the updateGrassHopper function as update with setting the default values for some of the args.
toolbox.register("evaluate", benchmarks.ackley) #the function used to calculate the fitness set and evaluate.

def main():
    Swarm = toolbox.swarm(n = 10) #intializing the swarm with n agents.
    
    GEN = 1400 #the numer of max iterations.
    best = None #initializing the best as none.
    #print(Swarm)


    for agent in Swarm: #finding the best position for the initial swarm/population
        if not best or best.fitness > agent.fitness:
            best = creator.Agent(agent)
            best.fitness = agent.fitness

    #print(best)

    for g in range(GEN): #runing the GAO Gen times
        c = compute_c(g + 1, GEN) #computing c 
        for agent in Swarm: #updating each agent using the best solution found 
            toolbox.update(agent = agent, Population = Swarm, c = c, best = best )
        for agent in Swarm: #updating the best solution after updating the whole swarm.
            if not best or best.fitness > agent.fitness:
                best = creator.Agent(agent)
                best.fitness = agent.fitness
    #print(best)
        

if __name__ == "__main__":
    main()


