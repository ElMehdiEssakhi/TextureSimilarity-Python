import matplotlib.pyplot as plt
import numpy as np
import thisModule as tm
import descriptors as ds

def distance_pixel(query,filename='image_database/',setSize=66,resultNumber=8):
    if(len(query.shape)==3): #to add rgba condition later
        queryVect=tm.imageToVector(tm.normalizeImage(tm.rgb2gray(query)))
    else:
        queryVect=tm.imageToVector(tm.normalizeImage(query))
    D=[]
    for i in range (1,setSize+1):
        title=filename+str(i)+'.jpg'
        x=tm.imageToVector(tm.normalizeImage(tm.rgb2gray(plt.imread(title))))
        dist=tm.EuclideanDistance(queryVect,x)
        couple=[i,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def distance_histogram(query,images,setSize=66,resultNumber=8): #images as a matrix
    if len(query.shape)==3:
        gray=tm.rgb2gray(query)
        queryHist=tm.ds.histo(gray)/gray.size
    else:
        queryHist=tm.ds.histo(query)/query.size
    D=[]
    for i in range (0,setSize):
        thisHist=tm.ds.histo(images[:,:,i])/images[:,:,i].size
        dist=tm.EuclideanDistance(thisHist,queryHist)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def distance_variance(query,images,setSize=66,resultNumber=8):
    queryVar=ds.variance(tm.rgb2gray(query))
    D=[]
    for i in range (0,setSize):
        thisVar=ds.variance(images[:,:,i])
        dist=tm.EuclideanDistance(thisVar,queryVar)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def distance_energie(query,images,setSize=66,resultNumber=8):
    queryEnergie=ds.energie(tm.rgb2gray(query))
    D=[]
    for i in range (0,setSize):
        thisEnergie=ds.energie(images[:,:,i])
        dist=tm.EuclideanDistance(thisEnergie,queryEnergie)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def disatance_entropie(query,images,setSize=66,resultNumber=8):
    queryEntropie=ds.entropie(tm.rgb2gray(query))
    D=[]
    for i in range (0,setSize):
        thisEntropie=ds.entropie(images[:,:,i])
        dist=tm.EuclideanDistance(thisEntropie,queryEntropie)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]       

def distance_contraste(query,images,setSize=66,resultNumber=8):
    q_norm = tm.normalizeImage(tm.rgb2gray(query))
    q_uint8 = (q_norm * 255).astype(np.uint8) 
    queryContraste = ds.contraste(q_uint8)
    
    D=[]
    for i in range (0,setSize):
        img_norm = tm.normalizeImage(images[:,:,i])
        img_uint8 = (img_norm * 255).astype(np.uint8)
        thisContraste=ds.contraste(img_uint8)

        dist=tm.EuclideanDistance(thisContraste,queryContraste)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def distance_homogenite(query,images,setSize=66,resultNumber=8):
    queryHomogenite=ds.homogenite(tm.rgb2gray(query))
    D=[]
    for i in range (0,setSize):
        thisHomogenite=ds.homogenite(images[:,:,i])
        dist=tm.EuclideanDistance(thisHomogenite,queryHomogenite)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def distance_average_color(query,images,setSize=66,resultNumber=8):
    queryAvgColor=ds.average_color(tm.normalizeImage(tm.rgb2gray(query)))
    D=[]
    for i in range (0,setSize):
        thisAvgColor=ds.average_color(tm.normalizeImage(images[:,:,i]))
        dist=tm.EuclideanDistance(thisAvgColor,queryAvgColor)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def distance_texture(query,images,setSize=66,resultNumber=8):
    queryTexture=tm.ds.texture(tm.normalizeImage(tm.rgb2gray(query)))
    D=[]
    for i in range (0,setSize):
        thisTexture=tm.ds.texture(tm.normalizeImage(images[:,:,i]))
        dist=tm.EuclideanDistance(thisTexture,queryTexture)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def distance_cooccurence(query,images,setSize=66,resultNumber=8):
    queryCooccurence=ds.cooccurence(tm.normalizeImage(tm.rgb2gray(query)*255).astype(np.uint8))
    D=[]
    for i in range (0,setSize):
        thisCooccurence=ds.cooccurence(tm.normalizeImage(images[:,:,i]*255).astype(np.uint8))
        dist=tm.EuclideanDistance(thisCooccurence.flatten(),queryCooccurence.flatten())
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]