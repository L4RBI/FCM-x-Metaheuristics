import os
from metaheuritics import metaheuristics
from indexes import *
from tools import *
from fuzzy import *


""" dir = ".\Images"
for subdir,dirs,files in os.walk(dir):
    for file in files:
        path = os.path.join(subdir,file)
        print("the image:",path )
        for i in range(3,10):
            print("centers : ",i)
            x= metaheuristics(path, i, m = 2)
            print("pso")
            z, time = x.pso(N = 100 , GEN = 2000)
            p = pc(membership = membership(Histogram(path),z,2))
            c = ce(membership(Histogram(path),z,2))
            s = sc(Histogram(path), z, 2)
            x = xb(Histogram(path),z,2)
            print("pc:",p,"ce:",c,"sc:",s,"xb",x)
            z, time = x.bat(N = 100 , GEN = 2000)
            p = pc(membership = membership(Histogram(path),z,2))
            c = ce(membership(Histogram(path),z,2))
            s = sc(Histogram(path), z, 2)
            x = xb(Histogram(path),z,2)
            print("pc:",p,"ce:",c,"sc:",s,"xb",x)
 
 """
path = "Images\Images\T07.JPG"
hist = Histogram(path)
C = 2
Initial_centers= numpy.array([[random.uniform(0,255) , random.uniform(0,numpy.amax(hist))] for _ in range(C)])
f = FuzzyCMeans(n_clusters = C, initial_centers = Initial_centers, histogram = hist, m = 2, max_iter= 2000)
centers,U = f.compute()
wow= metaheuristics(path, C, m = 2)
z = wow.gao(N = 50 , GEN = 2000)
print(centers)
print(z)
