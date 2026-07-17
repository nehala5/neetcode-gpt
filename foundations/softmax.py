import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
       # start by having numpy sorting function so we can get direct max integer
       maxi = np.max(z)
       # step 2 - subtract maxi from all elements - no need loop we can write dreictly
       # from this [1,2,3] to [-2, -1, 0]
       z = z- maxi
       # next step is expoenetially covert
       # [ -2, -1, 0] like this 
       z = np.exp(z)
       # now we have an array of exponential z values
       # now divide each element by total for it find sum first
       tot = np.sum(z)
       z = z/tot 
       # here the hwole array is divided at once 
       # then ;st step convert to 4 decimal format 
       return np.round(z,4)
