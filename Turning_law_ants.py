import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from numba import njit
import time


# Fonctions

@njit
def vectorToAngle(vector):
    
    nomarlizedVector = vector / np.sqrt(np.sum(vector**2))
    
    if nomarlizedVector[1] >= 0 :
        return np.arccos(nomarlizedVector[0])
        
    else :
        return np.arccos(nomarlizedVector[0]) + np.pi

    

@njit
def positionsToAngles(positionAnt, positionAntsSameEyesight, positionAntsOtherEyesight) :
    
    eyesightSameAnglesVector = np.zeros(len(positionAntsSameEyesight))
    eyesightOtherAnglesVector = np.zeros(len(positionAntsOtherEyesight))

    counter = 0
    
    for x in positionAntsSameEyesight:
        
        eyesightSameAnglesVector[counter] = vectorToAngle(x - positionAnt)
        counter += 1
        
    counter = 0
    
    for x in positionAntsOtherEyesight:
        
        eyesightOtherAnglesVector[counter] = vectorToAngle(x - positionAnt)
        counter += 1
        
    return np.sort(eyesightSameAnglesVector), np.sort(eyesightOtherAnglesVector)


@njit
def getNumberAntsInChunks(sorted_eyesightSameAnglesVector, sorted_eyesightOtherAnglesVector) :
    
    referenceAngle = sorted_eyesightOtherAnglesVector[0]
    chunks_values = sorted_eyesightOtherAnglesVector - referenceAngle
    weights_values = sorted_eyesightSameAnglesVector - referenceAngle
    
    numberChunks = len(chunks_values) 
    numberWeights = len(weights_values)
    
    numberAntsInChunks = np.zeros(numberChunks)
    
    counterChunks, counterWeights = 0, 0
    
    while counterChunks < numberChunks and counterWeights < numberWeights :
        
        while weights_values[counterWeights] <= chunks_values[counterChunks] :
            
            numberAntsInChunks[counterChunks-1] += 1
            
            counterWeights += 1
            
            if counterWeights >= numberWeights:
                break
            
            
        counterChunks += 1
         
    # Il faut ajouter le dernier chunk
    
    while counterWeights < numberWeights :
            
        if weights_values[counterWeights] > chunks_values[-1] :
            numberAntsInChunks[-1] += 1
        
            counterWeights += 1
    
    
    return numberAntsInChunks

@njit
def translateToMinusPiPiInterval(x): 
    
    if x > np.pi :
        
        return x - 2*np.pi
    
    elif x < -np.pi :
        
        return x + 2*np.pi

    else :
        return x

@njit
def getCentersChunks(sorted_eyesightOtherAnglesVector):
    
    numberChunks = len(sorted_eyesightOtherAnglesVector)
    centersChunks = np.zeros(numberChunks)
    
    centersChunks[-1] = np.pi + (sorted_eyesightOtherAnglesVector[-1] + sorted_eyesightOtherAnglesVector[0]) / 2
        
    centersChunks[:-1] = (sorted_eyesightOtherAnglesVector[:-1] + sorted_eyesightOtherAnglesVector[1:]) / 2
        
    return centersChunks

    
# Premiere approximation, a ameliorer
@njit
def computeWeightChunksDirection(centersChunks, directionAngle, sigmaDirection):
    
    '''
    Pour simplifier, on va regarder la distance du centre du chunk a la direction
    et mettre un poids comme cela
    '''
    
    distanceChunksDirection = centersChunks - directionAngle
    
    for numberChunk in range(len(distanceChunksDirection)) :
        
        distanceChunksDirection[numberChunk] = translateToMinusPiPiInterval(distanceChunksDirection[numberChunk])
    
    weightsChunksDirection = np.exp( - sigmaDirection * distanceChunksDirection**2)
    
    return weightsChunksDirection / np.sum(weightsChunksDirection)


@njit
def drawChunk(numberAntsInChunks, weightsChunksDirection, convexParam) :
    
    weightsChunksAnts = (numberAntsInChunks + 1) / np.sum(numberAntsInChunks)
    
    # Ou logique
  #  weightsChunks = (1 - convexParam) * weightsChunksAnts + convexParam * weightsChunksDirection
    # Et logique
    weightsChunks = weightsChunksAnts * (weightsChunksDirection/weightsChunksAnts)**convexParam
    
    repartitionFunctionChunks = np.cumsum(weightsChunks)
    repartitionFunctionChunks /= repartitionFunctionChunks[-1]
    
    uninfRandNumber = rd.rand()
    
    chunkNumber = 0
    
    while repartitionFunctionChunks[chunkNumber] < uninfRandNumber :
        chunkNumber += 1
    
    return chunkNumber



# Verifier la notion de centres de chunks


#%% Check that weights are well-computed 

sigmaDirection = .5
directionAngle = rd.rand() * 2 * np.pi
convexParam = .9

positionAnt = np.array([0,0])

positionAntsSameEyesight = 2*rd.rand(10,2) - 1

positionAntsOtherEyesight = 2*rd. rand(5,2) - 1


sorted_eyesightSameAnglesVector, sorted_eyesightOtherAnglesVector = positionsToAngles(positionAnt, positionAntsSameEyesight, positionAntsOtherEyesight)


numberAntsInChunks = getNumberAntsInChunks(sorted_eyesightSameAnglesVector, sorted_eyesightOtherAnglesVector) 

centersChunks = getCentersChunks(sorted_eyesightOtherAnglesVector)

weightsChunksDirection = computeWeightChunksDirection(centersChunks, directionAngle, sigmaDirection)

chunkNumber = drawChunk(numberAntsInChunks, weightsChunksDirection, convexParam)

plt.figure()
plt.scatter(sorted_eyesightSameAnglesVector,np.ones_like(sorted_eyesightSameAnglesVector))
plt.scatter(sorted_eyesightOtherAnglesVector,np.zeros_like(sorted_eyesightOtherAnglesVector))
#plt.scatter(centersChunks, numberAntsInChunks)
#plt.scatter(centersChunks, weightsChunksDirection)
plt.scatter([directionAngle], [-.25])
plt.scatter([centersChunks[chunkNumber]], [-.5])

























