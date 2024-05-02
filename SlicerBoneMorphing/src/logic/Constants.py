### RETURN VALUES ###
from src.logic.Enums import BcpdAccelerationMode, BcpdKernelType, BcpdNormalizationOptions, BcpdStandardKernel

EXIT_OK = 0
EXIT_FAILURE = 1

BCPD_MULTIPLE_VALUES_SEPARATOR = ","
"""
    Used when one parameter have multiple values, e.g. -G [string,real,file]
"""

VALUE_NODE_NOT_SELECTED = 0  # Via Slicer documentation

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
BCPD_DEFAULT_VALUE_CONVERGENCE_MIN_ITERATIONS = 1

## Normalization ##
BCPD_DEFAULT_VALUE_NORMALIZATION_OPTIONS = BcpdNormalizationOptions.X.value

### PREPROCESSING PARAMETERS ###
PREPROCESSING_KEY = "preprocessing"
PREPROCESSING_KEY_DOWNSAMPLING_DISTANCE_THRESHOLD = "ddt"
PREPROCESSING_KEY_NORMALS_ESTIMATION_RADIUS = "ner"
PREPROCESSING_KEY_FPFH_ESTIMATION_RADIUS = "fer"
PREPROCESSING_KEY_MAX_NN_NORMALS = "mnnn"
PREPROCESSING_KEY_MAX_NN_FPFH = "mnf"

PREPROCESSING_DEFAULT_VALUE_DOWNSAMPLING_DISTANCE_THRESHOLD = 0.05
PREPROCESSING_DEFAULT_VALUE_RADIUS_NORMAL_SCALE = 0.5
PREPROCESSING_DEFAULT_VALUE_RADIUS_FEATURE_SCALE = 10
PREPROCESSING_DEFAULT_VALUE_MAX_NN_NORMALS = 10
PREPROCESSING_DEFAULT_VALUE_MAX_NN_FPFH = 100

### REGISTRATION PARAMETERS ###
REGISTRATION_KEY_DISTANCE_THRESHOLD = "rdt"
REGISTRATION_KEY_FITNESS_THRESHOLD = "rft"
REGISTRATION_KEY_MAX_ITERATIONS = "rmi"
REGISTRATION_KEY_ICP_DISTANCE_THRESHOLD = "idt"

REGISTRATION_DEFAULT_VALUE_DISTANCE_THRESHOLD = 1
REGISTRATION_DEFAULT_VALUE_FITNESS_THRESHOLD = 0.999
REGISTRATION_DEFAULT_VALUE_MAX_ITERATIONS = 30
REGISTRATION_DEFAULT_VALUE_ICP_DISTANCE_THRESHOLD = 1

### POSTPROCESSING PARAMETERS ###
POSTPROCESSING_KEY = "postprocessing"
POSTPROCESSING_KEY_CLUSTERING_SCALING = "cs"
POSTPROCESSING_KEY_SMOOTHING_ITERATIONS = "sis"

POSTPROCESSING_DEFAULT_VALUE_CLUSTERING_SCALING = 1
POSTPROCESSING_DEFAULT_VALUE_SMOOTHING_ITERATIONS = 0

### MAX VALUES ###
BCPD_MAX_VALUE_KAPPA = 1000
