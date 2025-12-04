import numpy as np

def variance(image):
    return np.var(image)

def energie(I):     
    return np.sum(I**2)

def entropie(I):
    epsilon=1e-10
    I= np.where(I==0 ,epsilon,I)
    ent=np.sum(I*np.log2(I))
    return -ent

def contrast(I):
    # This measures how far pixels are from the mean
    return np.std(I)

def contrasteGLCM(I):
    c,l= I.shape
    s=0
    for i in range(c):
        for j in range(l):
            s+=(I[i,j])*(i-j)**2
    return s

def homogenite(I):
    l,c= I.shape
    mo=0
    for i in range(l):
        for j in range(c):
            mo+=I[i,j]/(1+abs(i-j))
    return mo

def average_color(img):
    return np.mean(img)

def histo(img):
    hitsreq,binsreq=np.histogram(img,256,[0, 256])
    return hitsreq

def texture(I):
    h=np.array([variance(I),entropie(I),energie(I),homogenite(I),contrast(I)])
    return h

def cooccurence(I):
    l,c=I.shape
    m=np.max(I)
    CC=np.zeros((255,255))
    for i in range(l):
        for j in range(c-1):
                CC[I[i][j]][I[i][(j)+1]]+=1 
    return CC