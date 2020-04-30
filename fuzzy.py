from tools import *

class FuzzyCMeans:
        def __init__(self, n_clusters, initial_centers , histogram , max_iter=250, m=2, error=1e-5 ):
            assert m > 1
            assert initial_centers.shape[0] == n_clusters
            self.U = None
            self.centers = initial_centers
            self.max_iter = max_iter
            self.m = m
            self.error = error
            self.histogram=histogram

        def membership(self, histogram, centers):
            U_temp = cdist( histogram , centers , 'euclidean')
            U_temp = numpy.power(U_temp,2/(self.m - 1))
            denominator_ = U_temp.reshape((histogram.shape[0], 1, -1)).repeat(U_temp.shape[-1], axis=1)
            denominator_ = U_temp[:, :, numpy.newaxis] / denominator_
            return 1 / denominator_.sum(2)

        def Centers(self,histogram,U):
            um = U ** self.m
            return (histogram.T @ um / numpy.sum(um, axis=0)).T

        def newImage(self,U,centers,im):
            best = numpy.argmax(self.U, axis = -1)
            print(best)
            image = im
            for i in range(256):
                image[image == i] = centers[best[i]][0]#image = numpy.where( image == float(i), centers[best[i]][0], image) 
            for i in range(256):
                image[image == i] = centers[best[i]][0]
            return image

        def compute(self):
            self.U = self.membership( self.histogram , self.centers)

            past_U = numpy.copy(self.U)

            for i in range(self.max_iter):

                self.centers = self.Centers( self.histogram , self.U)
                self.U = self.membership( self.histogram , self.centers)

                if norm(self.U - past_U) < self.error:
                    break
                past_U = numpy.copy(self.U)
            
            return self.centers, self.U

def main():
    path = ".\Images\Images\T07.JPG"

    Initial_centers= numpy.array([[random.uniform(0,255) , random.uniform(0,numpy.amax(Histogram(path)))] for _ in range(3)])

    f = FuzzyCMeans(n_clusters = 3, initial_centers = Initial_centers, histogram = Histogram(path))

    centers,U = f.compute()
    print(centers)

    """im = f.newImage(U,centers,image)
    print(centers)
    print(im)
    print(image)
    plt.imshow(im, cmap = 'gray')
    #plt.imshow(image, cmap = 'gray',interpolation='nearest')
    plt.show()"""

if __name__ == "__main__":
    main()