import numpy
from scipy.spatial.distance import cdist
from math import sqrt

def Dis(x,y):  
     dist = sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)  
     return dist 

"""def instraDistance(data,membership,centers):
    membership = numpy.argmax(membership, axis=-1)
    temp = []
    for i in range(len(membership)):
        temp.append(Dis(data[i],centers[membership[i]]))
    temp = numpy.power(temp, 2)
    return numpy.sum(temp)"""

#function that calculates the membership of each point based on the classic equation
def membership(histogram, centers):
    U_temp = cdist( histogram , centers , 'euclidean')
    U_temp = numpy.power(U_temp,2/(self.m - 1))
    denominator_ = U_temp.reshape((histogram.shape[0], 1, -1)).repeat(U_temp.shape[-1], axis=1)
    denominator_ = U_temp[:, :, numpy.newaxis] / denominator_
    return 1 / denominator_.sum(2)

def instraDistance(histogram,membership,centers):

    Membership = membership(histogram = histogram, centers = centers)
    Membership = numpy.argmax(Membership, axis=-1)
    index = 0 
    for center in centers:
        temp = []
        for i in range(len(Membership)):
            if Membership[i] == index:
                temp.append(Dis(histogram[i],center))
        temp = numpy.power(temp,2)
        center.fitness = numpy.sum(temp)
        index += 1
        





