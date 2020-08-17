import os
from metaheuritics import metaheuristics
from indexes import *
from tools import *
from fuzzy import *


dir = r"Images\brains"
for subdir, dirs, files in os.walk(dir):
    for file in files:
        path = os.path.join(subdir, file)
        C = 4
        x = metaheuristics(path, C, m=2)
        Initial_centers, time = x.pso(N=50, GEN=2000)
        f = FuzzyCMeans(n_clusters=C, initial_centers=Initial_centers,
                        histogram=Histogram(path), m=2, max_iter=250)
        centers, U = f.compute()
        p = pc(membership=U)
        c = ce(membership=U)
        s = sc(Histogram(path), centers, 2)
        x = xb(Histogram(path), centers, 2)
        print("the image:", path, "pc:", p, "ce:",
              c, "sc:", s, "xb:", x, "time:", time)


""" path = "Images\Images\T07.JPG"
hist = Histogram(path)
C = 5
wow = metaheuristics(path, C, m=2)
z, time = wow.pso(N=1, GEN=10)
_membership = membership(histogram=hist, centers=z, m=2)
Ni = numpy.bincount(numpy.argmax(_membership, axis=-1))
print(Ni)
Initial_centers = numpy.array(
    [[random.uniform(0, 255), random.uniform(0, numpy.amax(hist))] for _ in range(C)])
f = FuzzyCMeans(n_clusters=C, initial_centers=z,
                histogram=hist, m=2, max_iter=10)
centers, U = f.compute()
s = sc(hist, centers, 2)
Ni = numpy.bincount(numpy.argmax(U, axis=-1))
print(Ni)
print(s)
 """
