### RETURN VALUES ###
from src.logic.Enums import BcpdAccelerationMode, BcpdKernelType, BcpdNormalizationOptions, BcpdStandardKernel

EXIT_OK = 0
EXIT_FAILURE = 1

BCPD_MULTIPLE_VALUES_SEPARATOR = ","
"""
    Used when one parameter have multiple values, e.g. -G [string,real,file]
"""

VALUE_NODE_NOT_SELECTED = 0  # Via Slicer documentation

### PREPROCESSING PARAMETERS ###
PREPROCESSING_KEY = "preprocessing"
PREPROCESSING_KEY_VOXEL_SIZE_SCALING = "vss"
PREPROCESSING_RANSAC_MAX_ATTEMPTS = "rma"

###  BCPD PARAMETERS###
BCPD_KEY = "bcpd"

## Tuning ##
BCPD_VALUE_KEY_OMEGA = "-w"
BCPD_VALUE_KEY_LAMBDA = "-l"
BCPD_VALUE_KEY_BETA = "-b"
BCPD_VALUE_KEY_GAMMA = "-g"
BCPD_VALUE_KEY_KAPPA = "-k"

## Kernel ##
BCPD_VALUE_KEY_KERNEL = "-G"

## Acceleration ##
BCPD_VALUE_KEY_NYSTORM_G = "-K"
BCPD_VALUE_KEY_NYSTORM_P = "-J"
BCPD_VALUE_KEY_NYSTORM_R = "-r"

BCPD_VALUE_KEY_KD_TREE = "-p"
BCPD_VALUE_KEY_KD_TREE_SCALE = "-d"
BCPD_VALUE_KEY_KD_TREE_RADIUS = "-e"
BCPD_VALUE_KEY_KD_TREE_THRESHOLD = "-f"

## Downsampling ##
BCPD_VALUE_KEY_DOWNSAMPLING = "-D"

## Convergence ##
BCPD_VALUE_KEY_CONVERGENCE_TOLERANCE = "-c"
BCPD_VALUE_KEY_CONVERGENCE_MAX_ITERATIONS = "-n"
BCPD_VALUE_KEY_CONVERGENCE_MIN_ITERATIONS = "-N"

## Normalization ##
BCPD_VALUE_KEY_NORMALIZATION = "-u"

## Tuning parameters ##
BCPD_DEFAULT_VALUE_OMEGA = 0.1
BCPD_DEFAULT_VALUE_LAMBDA = 10
BCPD_DEFAULT_VALUE_BETA = 10
BCPD_DEFAULT_VALUE_GAMMA = 0.1
BCPD_DEFAULT_VALUE_KAPPA = 1000

## Kernel functions ##
BCPD_DEFAULT_VALUE_KERNEL_TYPE = BcpdKernelType.STANDARD.value
BCPD_DEFAULT_VALUE_STANDARD_KERNEL_TYPE = BcpdStandardKernel.G0.value
BCPD_DEFAULT_VALUE_TAU = 1
BCPD_DEFAULT_VALUE_INPUT_MESH_PATH = ""
BCPD_DEFAULT_VALUE_KERNEL_NEIGBOURS = 10
BCPD_DEFAULT_VALUE_KERNEL_NEIGHBOUR_RADIUS = 10
BCPD_DEFAULT_VALUE_KERNEL_BETA = 1
BCPD_DEFAULT_VALUE_KERNEL_K_TILDE = 1
BCPD_DEFAULT_VALUE_KERNEL_EPSILON = 1

## Acceleration mode ##
BCPD_DEFAULT_VALUE_ACCELERATION_MODE = BcpdAccelerationMode.AUTOMATIC.value
BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_G = 140
BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_J = 600
BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_R = 1
BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_SCALE = 7
BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_RADIUS = 0.3
BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_SIGMA_THRESHOLD = 0.3

## Downsampling ##
BCPD_DEFAULT_VALUE_DOWNSAMPLING_OPTIONS = "B,5000,0.08"

## Convergence ##
BCPD_DEFAULT_VALUE_CONVERGENCE_TOLERANCE = 0.000001
BCPD_DEFAULT_VALUE_CONVERGENCE_MAX_ITERATIONS = 1000
BCPD_DEFAULT_VALUE_CONVERGENCE_MIN_ITERATIONS = 30

## Normalization ##
BCPD_DEFAULT_VALUE_NORMALIZATION_OPTIONS = BcpdNormalizationOptions.X.value


RADIUS_NORMAL_SCALING = 4
RADIUS_FEATURE_SCALING = 10
MAX_NN_NORMALS = 30
MAX_NN_FPFH = 100
CLUSTERING_VOXEL_SCALING = 3
SMOOTHING_ITERATIONS = 2
FILTERING_ITEARTIONS = 100

### MAX VALUES ###
BCPD_MAX_VALUE_KAPPA = 1000
