import matplotlib.pyplot as plt
import thisModule as tm
#import descriptors as ds
import descriptorsV2 as desv2

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
        queryHist=desv2.histo(gray)/gray.size
    else:
        queryHist=desv2.histo(query)/query.size
    D=[]
    for i in range (0,setSize):
        thisHist=desv2.histo(images[:,:,i])/images[:,:,i].size
        dist=tm.EuclideanDistance(thisHist,queryHist)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def distance_variance(query,images,setSize=66,resultNumber=8):
    queryVar=desv2.variance(tm.rgb2gray(query))
    D=[]
    for i in range (0,setSize):
        thisVar=desv2.variance(images[:,:,i])
        dist=tm.EuclideanDistance(thisVar,queryVar)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def distance_energie_global(query,images,setSize=66,resultNumber=8):
    queryEnergie=desv2.energie_global(tm.rgb2gray(query))
    D=[]
    for i in range (0,setSize):
        thisEnergie=desv2.energie_global(images[:,:,i])
        dist=tm.EuclideanDistance(thisEnergie,queryEnergie)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def disatance_entropie(query,images,setSize=66,resultNumber=8):
    queryEntropie=desv2.entropie(tm.rgb2gray(query))
    D=[]
    for i in range (0,setSize):
        thisEntropie=desv2.entropie(images[:,:,i])
        dist=tm.EuclideanDistance(thisEntropie,queryEntropie)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]       

def distance_contraste_rms(query,images,setSize=66,resultNumber=8):
    queryContraste = desv2.contrast_rms(tm.rgb2gray(query))
    D=[]
    for i in range (0,setSize):
        thisContraste=desv2.contrast_rms(images[:,:,i])

        dist=tm.EuclideanDistance(thisContraste,queryContraste)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def distance_average_color(query,images,setSize=66,resultNumber=8):
    queryAvgColor=desv2.average_color(tm.rgb2gray(query))
    D=[]
    for i in range (0,setSize):
        thisAvgColor=desv2.average_color(images[:,:,i])
        dist=tm.EuclideanDistance(thisAvgColor,queryAvgColor)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def distance_contrast_glcm(query,images,setSize=66,resultNumber=8):
    cooc=desv2.cooccurence(tm.rgb2gray(query))
    queryContrast=desv2.contrast_glcm(cooc)
    D=[]
    for i in range (0,setSize):
        thisCooc=desv2.cooccurence(images[:,:,i])
        thisContrast=desv2.contrast_glcm(thisCooc)
        dist=tm.EuclideanDistance(queryContrast,thisContrast)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def distance_homogenite_glcm(query,images,setSize=66,resultNumber=8):
    cooc=desv2.cooccurence(tm.rgb2gray(query))
    queryHomogenite=desv2.homogenite_glcm(cooc)
    D=[]
    for i in range (0,setSize):
        thisCooc=desv2.cooccurence(images[:,:,i])
        thisHomogenite=desv2.homogenite_glcm(thisCooc)
        dist=tm.EuclideanDistance(thisHomogenite,queryHomogenite)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def distance_cooccurence(query,images,setSize=66,resultNumber=8):
    queryCooccurence=desv2.cooccurence(tm.rgb2gray(query))
    D=[]
    for i in range (0,setSize):
        thisCooccurence=desv2.cooccurence(images[:,:,i])
        dist=tm.EuclideanDistance(thisCooccurence.flatten(),queryCooccurence.flatten())
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def distance_texture(query,images,setSize=66,resultNumber=8):
    queryTexture=desv2.get_final_vector(tm.rgb2gray(query))
    D=[]
    for i in range (0,setSize):
        thisTexture=desv2.get_final_vector(images[:,:,i])
        dist=tm.EuclideanDistance(thisTexture,queryTexture)
        couple=[i+1,dist]
        D.append(couple)
    return sorted(D, key=lambda item: item[1])[:resultNumber]

def distance_texture_precomputed(imgId, precomputed_features,resultNumber=8):
    img_features = precomputed_features[imgId-1]
    distances = []
    for i,features in enumerate(precomputed_features):
        dist = tm.EuclideanDistance(img_features, features)
        distances.append((i+1, dist))
    return sorted(distances, key=lambda item: item[1])[:resultNumber]