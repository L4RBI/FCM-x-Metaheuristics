from fuzzy import *


path = ".\Images\Images\\T07.JPG"
image = imread(path)
if len(image.shape) != 2:
    gray = lambda rgb : numpy.dot(rgb[... , :3] , [0.2989, 0.5870, 0.1140]) 
    gray = gray(image) 
    image = gray
hist,bins = numpy.histogram(image.ravel(),256,[0,256])
#histogram = Histogram(path)
C = 5
Initial_centers= numpy.array([[random.uniform(0,255) , random.uniform(0,numpy.amax(hist))] for _ in range(C)])
#Initial_centers = [[  64.80076321, 2938.94607211], [  89.02764696, 1280.58068295], [ 117.92091512, 1782.9144404 ], [182.90076514, 465.3391341 ], [237.77699924,  22.65330623], [ 128.01069005, 2194.42390433],[79.41082366, 872.0583745 ]]
f = FuzzyCMeans(n_clusters = C, initial_centers = Initial_centers, histogram = adapt(hist), m = 2)
centers,U = f.compute()

M = membership(adapt(hist),centers,2)
if numpy.array_equal(M,U):
    print("gg")
    best = numpy.argmax(M, axis = -1)
    print(numpy.bincount(best))
""" im = f.newImage(U,centers,image)
plt.imshow(im, cmap = 'gray')
plt.show() """

#print("centers:",centers,"J function: ",J(adapt(hist),U,centers,2))