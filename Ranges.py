import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from numba import njit
import time


# Parameters

nbAntsL = 13
numberAntsRtoL = 10

lengthHorizontal = 2
lengthVertical = 1

maxDistVisu = .7

radius = .2

# Functions

@njit
def distance(position1,position2):
    
    return np.sqrt(np.sum((position1-position2)**2))

@njit
def distance_x(position1,position2):
    
    return position2[0]-position1[0]

@njit
def testBBlocksCfromA(A,B,C,r):
    
    x1, y1 = B-A
    x2, y2 = C-A
    
    norme1, norme2 = np.sqrt(x1**2 + y1**2), np.sqrt(x2**2 + y2**2)
    
    
    if C[1] >= B[1] :
      
        if y1 - x1 * r / norme1 <= y2 + x2 * r / norme2 :
            
            return False
        
        else:
            
            return True
        
    else:
        
        if y1 + x1 * r / norme1 <= y2 - x2 * r / norme2 :
            
            return False
        
        else:
            
            return True


# On pourrait aussi considerer une anisotropie en x et y

#@njit
#def distanceAnisotropic(position1,position2,coeffAnisotropic):
#    
#    return * np.sqrt(2 * (1 - coeffAnisotropic) * (position1[0] - position2[0])**2 + 2 * coeffAnisotropic * (position1[1] - position2[1])**2)

#%%

positions2DLtoR = rd.rand(nbAntsL,2)
positions2DLtoR[:,0] *= lengthHorizontal
positions2DLtoR[:,1] *= lengthVertical

positions2DRtoL = rd.rand(numberAntsRtoL,2)
positions2DRtoL[:,0] *= lengthHorizontal
positions2DRtoL[:,1] *= lengthVertical

#%%


# Sort ants LtoR from R to L and ants RtoL from L to R

sortPositL = positions2DLtoR[np.argsort(positions2DLtoR[:,0])][::-1]
sorted_positions2DRtoL = positions2DRtoL[np.argsort(positions2DRtoL[:,0])]



# We update the ants's state from objective to origine

# The first ant has nothing in its visual range nor in its antenna contact range

# We can therefore easily compute its new state.


plt.figure()
plt.scatter(positions2DLtoR[:,0],positions2DLtoR[:,1])

oldVisuRangeL = np.array([]).astype(np.int64)

for i in range(1,nbAntsL):
    
    nbAntsRange = 0
    
    if distance_x(sortPositL[i],sortPositL[i-1]) < maxDistVisu:
        nbAntsRange += 1
    
    for k in oldVisuRangeL:
        
        if distance_x(sortPositL[i],sortPositL[k]) < maxDistVisu:
            nbAntsRange += 1
        
        else:
            break
    
    VisuRangeL = np.zeros(nbAntsRange).astype(np.int64)
    
    counterVisuRangeL = 0
    
    if distance_x(sortPositL[i],sortPositL[i-1]) < maxDistVisu:
        
        VisuRangeL[counterVisuRangeL] = i-1
        counterVisuRangeL += 1
    
    for k in oldVisuRangeL:
        
        if distance_x(sortPositL[i],sortPositL[k]) < maxDistVisu:
            VisuRangeL[counterVisuRangeL] = k
            counterVisuRangeL += 1
        
        else:
            break

    oldVisuRangeL = VisuRangeL

    actualVisu = VisuRangeL.copy()
    
   # if len(actualVisu) > 0:
    
      #  for j in range(len(actualVisu)):
            
     #       k = actualVisu[j]
            
    #        if  distance(sortPositL[i],sortPositL[k]) > maxDistVisu:
                
   #             actualVisu[j] = -1

       # if len(np.nonzero(actualVisu+1)[0]) > 0:
            
       #     actualVisu = actualVisu[np.nonzero(actualVisu+1)[0]]-1
            
    plt.figure()
    plt.scatter(positions2DLtoR[:,0],positions2DLtoR[:,1], color = 'blue')
    plt.scatter(sortPositL[i,0], sortPositL[i,1], color = 'green')
        
    for n in range(len(actualVisu)):
        
        isIndexBlocked = False
        
        for m in range(len(actualVisu)):
            
           # isIndexBlocked = (isIndexBlocked or testBBlocksCfromA(sortPositL[i],sortPositL[actualVisu[n]],sortPositL[actualVisu[m]], radius))
          
            if testBBlocksCfromA(sortPositL[i],sortPositL[actualVisu[m]],sortPositL[actualVisu[n]], radius) :
              
                plt.scatter(sortPositL[actualVisu[m],0], sortPositL[actualVisu[m],1], color = 'red')
            else :
                plt.scatter(sortPositL[actualVisu[m],0], sortPositL[actualVisu[m],1], color = 'orange')
          
       # if not isIndexBlocked :
       #     plt.scatter(sortPositL[actualVisu[n],0], sortPositL[actualVisu[n],1], color = 'red')
    
   # plt.savefig('Figures/figure' + str(i) + '.pdf')

   # print(actualVisu)


#####






















