from enum import Enum

class BcpdKernelType(Enum):
    STANDARD = 0 
    GEODESIC = 1 

class BcpdStandardKernel(Enum):
    G0 = 0
    G1 = 1
    G2 = 2
    G3 = 3

class BcpdNormalizationOptions(Enum):
    E = 0
    X = 1 
    Y = 2
    N = 3

class BcpdAccelerationMode(Enum):
    AUTOMATIC = 0
    MANUAL = 1 
