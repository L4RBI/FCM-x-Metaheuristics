from tools import *


from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

creator.create("Fitness", base.Fitness, weights=(-1.0,)) #creating an object Fitness with deap.
creator.create("Bat", list , fitness = creator.Fitness, velocity = list , frequency = None, rate = None, loudness = None, init_rate = None, best = None) #creating an object Agent with deap.

#___________________________________________________________________________

path = "..\Images\Images\T07.JPG"
histogram = Histogram(path)
    
#____________________________________________________________________________



#the function that generates the initial an Bat randomly.
def generate(bmin, bmax, dim, size):
    bat = creator.Bat([random.uniform(bmin,bmax), random.uniform(bmin,numpy.amax(histogram))] for _ in range(size))
    bat.init_rate = random.uniform(0,0.95)
    bat.rate = bat.init_rate
    bat.loudness = 1
    bat.velocity = [[0,0] for _ in range(size)]
    bat.fitness = toolbox.evaluate(data = histogram , bat = bat)
    bat.best = creator.Bat(bat)
    bat.best.fitness = bat.fitness
    return bat


#updates the position of the agent using the 2, 3 and 4 equations of the paper.
def updateBat(bat, bmin, bmax, best, fmin, fmax, A, G, dim, size, alpha = 0.8, gamma = 0.9):
    
    """bat.frequency = fmin + (fmax - fmin) * random.uniform(0,1)
    bat.velocity = list(map(operator.add, bat.velocity, (_ * bat.frequency for _ in map(operator.sub, bat, best))))
    solution = creator.Bat(list(map(operator.add, bat , bat.velocity)))"""
    
    rand = random.uniform(0,1)#numpy.random.random_sample()
    #print("best random: ",rand,"the rate copared to it: ",bat.rate)
    if rand > bat.rate: #random walk using equation 5 from the paper on the global best solution.
        eps = random.uniform(-1,1)
        eps_A = eps * A 
        solution = numpy.add(best,eps_A)#( _ + eps_A for _ in best)
        solution = creator.Bat(list(solution)) 

        """print("-around the best-")
        print("eps: ",eps)
        print("A: ",A)
        print("solution: ",solution)"""
    else:
        bat.frequency = [[fmin + (fmax - fmin) * random.uniform(0,1) for _ in range(dim)] for _ in range(size)] #equation 2 from the paper.
        dis = numpy.multiply(numpy.subtract(bat, best), bat.frequency)
        bat.velocity = numpy.add(bat.velocity, dis) #eauqtion 3 from the paper.
        solution = creator.Bat(numpy.add(bat , bat.velocity)) #equation 4 from the paper

        """print("-random walk-")
        print("frequency: ",bat.frequency)
        print("velocity: ",bat.velocity)
        print("solution: ",solution)"""

    for _ in solution: #making sure the bat doesn't go too far and stays in the objective function domain.
        for __ in range(len(_)):   
            if _[__] > bmax:
                _[__]  = bmax
            if _[__]  < bmin:
                _[__] = bmin
    #print("solution after boudning: ",solution)
    solution.fitness = toolbox.evaluate(data = histogram , bat = solution)#calculation the fitness of the Bat.

    rand = random.uniform(0,1)#numpy.random.random_sample()
    #print("best random: ",rand,"the loudness copared to it: ",bat.loudness,rand < bat.loudness)

    if bat.fitness > solution.fitness and rand < bat.loudness: #asserting the solution only if the solution is better and the bat iss too loud.
        #print("-asserting the solution-  rand:",rand,"loudness:",bat.loudness)
        bat[:] = solution
        bat.fitness = solution.fitness
        bat.loudness = alpha * bat.loudness
        bat.rate = bat.init_rate * (1 - math.exp(-alpha * (G+1)))
        #print("new loudness",bat.loudness)
        #print("new rate",bat.rate)

    if bat.best.fitness > bat.fitness: #updating the personal best.
        #print("pbest updated")
        bat.best = creator.Bat(bat)
        bat.best.fitness = bat.fitness

    if best.fitness > solution.fitness: #updating the global best.
        #print("best updated")
        best[:] = creator.Bat(solution)
        best.fitness = solution.fitness


def Evaluate(data, bat, m):
    M = membership(data, centers = bat, m = m)
    return (J(data, M, bat, m),)


toolbox = base.Toolbox()
toolbox.register("bat", generate, bmin=0, bmax = 255, dim = 2, size = 3)
toolbox.register("population", tools.initRepeat, list, toolbox.bat)
toolbox.register("update", updateBat, fmin = 0, fmax = 0.4, bmin = 0, bmax = 255, dim = 2, size = 3)
toolbox.register("evaluate", Evaluate, m = 2)



def main():
    Population = toolbox.population(n=20)
    """print(Population)
    print(" ")"""

    GEN = 1500
    best = None
    A = 0

    for bat in Population: #finding the initial global best
        if not best or best.fitness > bat.fitness:
            best = creator.Bat(bat)
            best.fitness = bat.fitness
        A += bat.loudness
        """print("the bat: ",bat , "the fitness: ", bat.fitness,"the loudness: ",bat.loudness,"the rate: ",bat.rate,"initrate",bat.init_rate,  )
        print(" ")"""
    mean_A = A / len(Population) #calculating the average loudness to use in equation 5 of the paper.
            
    for G in range(GEN): #computing the BA GEN times
        A=0
        """print("the best:", best, "fitness: ", best.fitness)
        print(" ")
        print("average loudness:",mean_A)"""
        for bat in Population: #updating the Bat postion one by one and opdating the Global best as well.
            toolbox.update(bat, best = best, A = mean_A, G = G)
            A += bat.loudness
            """print("the bat: ",bat , "the fitness: ", bat.fitness,"the loudness: ",bat.loudness,"the rate: ",bat.rate,"initrate",bat.init_rate,  )
            print(" ")"""
        mean_A = A / len(Population) #calculating the new average loudness

        """for bat in Population:
            if best.fitness > bat.fitness:
                best = creator.Bat(bat)
                best.fitness = bat.fitness"""
        print(best, best.fitness)
        

if __name__ == "__main__":
    main()


