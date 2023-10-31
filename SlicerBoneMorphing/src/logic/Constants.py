### RETURN VALUES ###
from src.logic.Enums import BCPDAccelerationMode, BCPDKernelMode, BCPDNormalizationOptions, BCPDStandardKernelMode

EXIT_OK = 0
EXIT_FAILURE = 1

###  BCPD OPTIONS ###
BCPD_MODE_KEY = "mode"
BCPD_OUTLIER_PROBABILITY_KEY = "w"
BCPD_DVL_KEY = "l"
BCPD_SMOOTH_RANGE_KEY = "b"
BCPD_POINT_MATCHING_RANDOMNESS_KEY = "g"
BCPD_MIXING_RANDOMNESS_KEY = "k"

### BCPD_PARAMETERS_DEFAULT_VALUES ###
## Tuning parameters ##
BCPD_DEFAULT_VALUE_OMEGA = 0.1
BCPD_DEFAULT_VALUE_LAMBDA = 10
BCPD_DEFAULT_VALUE_BETA = 10
BCPD_DEFAULT_VALUE_GAMMA = 0.1
BCPD_DEFAULT_VALUE_KAPPA = 1000

## Kernel functions ##
BCPD_DEFAULT_VALUE_KERNEL_TYPE = BCPDKernelMode.STANDARD.value
BCPD_DEFAULT_VALUE_STANDARD_KERNEL_TYPE = BCPDStandardKernelMode.G1.value
BCPD_DEFAULT_VALUE_TAU = 1
BCPD_DEFAULT_VALUE_INPUT_MESH_PATH = ""
BCPD_DEFAULT_VALUE_KERNEL_NEIGBOURS = 10
BCPD_DEFAULT_VALUE_KERNEL_NEIGHBOUR_RADIUS = 10
BCPD_DEFAULT_VALUE_KERNEL_BETA = 1
BCPD_DEFAULT_VALUE_KERNEL_K_TILDE = 1
BCPD_DEFAULT_VALUE_KERNEL_EPSILON = 1

## Acceleration mode ##
BCPD_DEFAULT_VALUE_ACCELERATION_MODE = BCPDAccelerationMode.AUTOMATIC.value
BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_G = 140
BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_J = 600 
BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_R = 1
BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_SCALE = 7
BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_RADIUS = 0.3
BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_SIGMA_THRESHOLD = 0.3 

## Downsampling ##
BCPD_DEFAULT_VALUE_DOWNSAMPLING_OPTIONS = "-DB,5000,0.08"

## Convergence ##
BCPD_DEFAULT_VALUE_CONVERGENCE_TOLERANCE = 0.000001
BCPD_DEFAULT_VALUE_CONVERGENCE_MAX_ITERATIONS = 1000
BCPD_DEFAULT_VALUE_CONVERGENCE_MIN_ITERATIONS = 30

## Normalization ##
BCPD_DEFAULT_VALUE_NORMALIZATION_OPTIONS = BCPDNormalizationOptions.X.value
