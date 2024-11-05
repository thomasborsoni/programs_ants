

@njit
def drawChunk(weightsChunks, sorted_eyesightOtherAnglesVector, directionAngle, sigma):
    
    numberWeightsPlusOne = np.sum(weightsChunks + 1)
    numberChunks = len(weightsChunks)
    
    nomalizedCumsumWeights = np.cumsum(weightsChunks + 1) / numberWeightsPlusOne
    
    securityCounter = 0
    
    while securityCounter < 100000 :
        
        # First, we draw a chunk
        uninfRandNumber = rd.rand()
        
        chunkNumber = 0
        
        while nomalizedCumsumWeights[chunkNumber] < uninfRandNumber :
            chunkNumber += 1
            
        if chunkNumber == numberChunks-1 :
            
            centerChunk = np.pi + sorted_eyesightOtherAnglesVector[-1] / 2
            
        else :
            
            centerChunk = (sorted_eyesightOtherAnglesVector[chunkNumber] + sorted_eyesightOtherAnglesVector[chunkNumber + 1]) / 2
            
        # We do the acceptance / rejection step according to the direction angle
        
        distanceChunkToDirection = np.abs(translateToMinusPiPiInterval(centerChunk - directionAngle))
        
        if rd.rand() < np.exp(-sigma * distanceChunkToDirection): # accept
        
            return chunkNumber, securityCounter
    
        else : # reject, try another chunk
            
            securityCounter += 1
     
