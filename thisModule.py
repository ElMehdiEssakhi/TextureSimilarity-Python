import matplotlib.pyplot as plt
import numpy as np
import descriptors as ds


def ImageAsArrayNum(index):
    Image=plt.imread(f'image_database/{index}.jpg')#not suted for png images
    return Image

def ImageAsArray(name,foldername):  
    Image=plt.imread(f'{foldername}/{name}')
    return Image

def rgb2gray(I): #not suted for png images
    array=np.array(I)
    gray=0.2989*array[0:,0:,0]+0.5870*array[0:,0:,1]+0.1140*array[0:,0:,2]
    return gray.astype(np.uint8)

def normalizeImage(I):
    Imin=I.min()
    Imax=I.max()
    Inorm=(I-Imin)/(Imax-Imin)
    return Inorm

def imageToVector(I:np.ndarray)->np.ndarray:
    return I.flatten()

def histo(img):
    hitsreq,binsreq=np.histogram(img,256,[0, 256])
    return hitsreq

def plotFirstFromEachCategory(categoriesCount,CategorySize,lines=2,columns=3):
    image=1
    plt.figure()
    for x in range (1,categoriesCount+1):
        I=plt.imread(f'image_database/{image}.jpg')
        plt.subplot(lines,columns,x)
        plt.imshow(I)
        plt.title(f'id={image}')
        plt.axis('off')
        image+=CategorySize


def EuclideanDistance(x,y):
    return np.sqrt(np.sum((x-y)**2))


def plotResults(Dsorted,filename='image_database/',columns=4,resultNumber=8):
    plt.figure()
    for i in range (resultNumber):
        index=Dsorted[i][0]
        path=f"{filename}{index}.jpg"
        plt.subplot(resultNumber//columns,columns,i+1)
        plt.title(f'{index}')
        plt.imshow(plt.imread(path))
        plt.axis('off')

def getImages(nbr_im, filename='image_database/'):
    I=plt.imread(f'{filename}1.jpg')
    nl,nc=rgb2gray(I).shape
    images=np.zeros((nl,nc,nbr_im)).astype(np.uint8)
    for i in range(nbr_im):
        images[:,:,i]=rgb2gray(plt.imread(f'{filename}{i+1}.jpg'))   
    return images

