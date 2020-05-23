from tools import *


from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

creator.create("Fitness", base.Fitness, weights=(-1.0,)) #creating an object Fitness with deap.
creator.create("Bat", list , fitness = creator.Fitness, velocity = list , frequency = None, rate = None, loudness = None, init_rate = None, best = None) #creating an object Agent with deap.
init_R = 0.5
init_A = 0.95

#___________________________________________________________________________

path = "..\Images\Images\T07.JPG"
histogram = Histogram(path)
size = 4  
dim = 2  
m = 0
M = 255
#____________________________________________________________________________


#the function that generates the initial an Bat randomly.
def generate(bmin, bmax):
    bat = creator.Bat([[random.uniform(bmin,bmax), random.uniform(bmin,numpy.amax(histogram))] for _ in range(size)])
    bat.init_rate = random.uniform(0,init_R)
    bat.rate = bat.init_rate
    bat.loudness = random.uniform(init_A , 2)
    bat.velocity = [[0, 0] for _ in range(size)]
    bat.fitness = toolbox.evaluate(bat)
    return bat


#updates the position of the agent using the 2, 3 and 4 equations of the paper.
def updateBat(bat, best, fmin, fmax, G, A, alpha = 0.9, gamma = 0.9):
    bat.frequency = fmin + (fmax - fmin) * numpy.random.uniform(0,1)
    dis = numpy.multiply(numpy.subtract(bat, best), bat.frequency)
    bat.velocity = list(numpy.add(bat.velocity, dis)) #eauqtion 3 from the paper.
    solution = creator.Bat(list(numpy.add( bat , bat.velocity))) #equation 4 from the paper
    # print("the bat:",bat)
    # print("the velocity:",bat.velocity)
    # print("the solution",solution)
    # print(" ")

    rand = numpy.random.random_sample()
    if rand > bat.rate : #random walk using equation 5 from the paper on the global best solution.
        # print("random walk random:", rand)
        # print(" ")
        solution = numpy.add(best , [[ random.uniform(-1,1) *A for _ in range(dim)] for _ in range(size)])
        #solution = numpy.add(best , numpy.multiply( A ,[random.gauss(0,1), random.gauss(0,1)]))
        solution = creator.Bat(list(solution)) 
        # print(solution)
        # print(" ")


    """ for _ in range(len(solution)): #making sure the bat doesn't go too far and stays in the objective function domain.
        if solution[_] > bmax:
            solution[_]  = bmax
        if solution[_]  < bmin:
            solution[_] = bmin
    """
    solution.fitness = toolbox.evaluate(solution) #calculation the fitness of the Bat.

    rand =numpy.random.random_sample()
    if bat.fitness >= solution.fitness and rand < A : #asserting the solution only if the solution is better and the bat iss too loud.
        # print("-asserting the solution-  rand:",rand)
        # print("bat: ",bat.fitness,"sol: ",solution.fitness)
        # print(" ")
        bat[:] = solution
        bat.fitness = solution.fitness
        bat.loudness = alpha * bat.loudness
        bat.rate = bat.init_rate * (1 - math.exp(-gamma * (G+1)))

    if best.fitness > solution.fitness: #updating the global best.
        # print("best updated")
        # print(" ")
        best[:] = creator.Bat(solution)
        best.fitness = solution.fitness
        # print(best)       
    # print(" ")
    # print(" ")

def Evaluate(bat, m):
    M = membership(histogram, centers = bat, m = m)
    return (J(histogram, M, bat, m),)

toolbox = base.Toolbox()
toolbox.register("bat", generate, bmin= m, bmax = M)
toolbox.register("population", tools.initRepeat, list, toolbox.bat)
toolbox.register("update", updateBat, fmin = 0, fmax = 0.2)
toolbox.register("evaluate", Evaluate, m = 2)





def main():
    N = 100
    Population = toolbox.population(n=N)
    mean_A = 0
    A = 0
    GEN = 2000
    best = None
    for bat in Population: #finding the initial global best
        if not best or best.fitness > bat.fitness:
            best = creator.Bat(bat)
            best.fitness = bat.fitness
        A += bat.loudness
        """print("the bat: ",bat , "the fitness: ", bat.fitness,"the loudness: ",bat.loudness,"the rate: ",bat.rate,"initrate",bat.init_rate,  )
        print(" ")"""
    print(best)        
    mean_A = A / N
    for G in range(GEN): #computing the BA GEN times
        # print(Population)
        # print(best, best.fitness)
        # print(" ")
        A = 0
        for bat in Population: #updating the Bat postion one by one and opdating the Global best as well.
            toolbox.update(bat, best = best, G = G, A = mean_A)
            A += bat.loudness
            """print("the bat: ",bat , "the fitness: ", bat.fitness,"the loudness: ",bat.loudness,"the rate: ",bat.rate,"initrate",bat.init_rate,  )
            print(" ")"""
        mean_A = A / N
    print(best,best.fitness)
    
if __name__ == "__main__":
    main()


