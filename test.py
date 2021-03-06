from fuzzy import *


path = "pic.jpg"
image = imread(path)
if len(image.shape) != 2:
    gray = lambda rgb : numpy.dot(rgb[... , :3] , [0.2989, 0.5870, 0.1140]) 
    gray = gray(image) 
    image = gray
hist,bins = numpy.histogram(image.ravel(),256,[0,256])
#histogram = Histogram(path)
#plt.imshow(image, cmap = 'gray')
#plt.show()
C = 7
#plt.plot(bins[:256],hist)
#plt.show()
Initial_centers= numpy.array([[random.uniform(0,255) , random.uniform(0,numpy.amax(hist))] for _ in range(C)])
#Initial_centers = [[  64.80076321, 2938.94607211], [  89.02764696, 1280.58068295], [ 117.92091512, 1782.9144404 ], [182.90076514, 465.3391341 ], [237.77699924,  22.65330623], [ 128.01069005, 2194.42390433],[79.41082366, 872.0583745 ]]
f = FuzzyCMeans(n_clusters = C, initial_centers = Initial_centers, histogram = adapt(hist), m = 2, max_iter= 2000)
centers,U = f.compute()
im = f.newImage(U,centers,image)
segs = []
for i in centers:
    print(i[0])
    segs.append(numpy.where( im == i[0], i[0], 0))
plt.figure()
for i in range(len(segs)):
    plt.subplot(1,C,i+1)
    plt.imshow(segs[i] , cmap = 'gray')
plt.show()

#print("centers:",centers,"J function: ",J(adapt(hist),U,centers,2))