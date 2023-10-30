from enum import Enum

class BCPDKernelMode(Enum):
    STANDARD = 0 
    GEODESIC = 1 

class BCPDStandardKernelMode(Enum):
    G0 = 0
    G1 = 1
    G2 = 2
    G3 = 3

class BCPDNormalizationOptions(Enum):
    E = 0
    X = 1 
    Y = 2
    N = 3

class BCPDAccelerationMode(Enum):
    AUTOMATIC = 0
    MANUAL = 1 
