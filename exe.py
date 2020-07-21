import os
from metaheuritics import metaheuristics
from indexes import *
from tools import *

dir = ".\Images"
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
 

