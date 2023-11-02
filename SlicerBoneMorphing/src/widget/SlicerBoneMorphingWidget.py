from src.logic.Constants import *
import ctk
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleWidget
from src.logic.SlicerBoneMorphingLogic import SlicerBoneMorphingLogic

from src.logic.Enums import BcpdAccelerationMode, BcpdKernelType, BcpdNormalizationOptions, BcpdStandardKernel
from qt import QComboBox

import os


class SlicerBoneMorphingWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        """Called when the application opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.bcpd_options = {}

    def setup(self):
        """Called when the application opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer)
        self.uiWidget = slicer.util.loadUI(
            self.resourcePath("UI/SlicerBoneMorphing.ui"))
        self.layout.addWidget(self.uiWidget)
        self.ui = slicer.util.childWidgetVariables(self.uiWidget)
        self.logic = SlicerBoneMorphingLogic(self)

        self.setup_ui()

    def setup_ui(self):
        """
            Method that sets up all UI elements and their dependencies
        """

        self.ui.sourceNodeSelectionBox.setMRMLScene(slicer.mrmlScene)
        self.ui.targetNodeSelectionBox.setMRMLScene(slicer.mrmlScene)

        self.ui.bcpdAdvancedControlsGroupBox.setVisible(False)

        self.setup_combo_box(self.ui.bcpdKernelTypeComboBox, BcpdKernelType,
                             self.show_kernel_type)
        self.ui.bcpdGeodesicKernelGroupBox.setVisible(False)

        self.setup_combo_box(self.ui.bcpdStandardKernelComboBox,
                             BcpdStandardKernel, None)

        self.setup_combo_box(self.ui.bcpdAccelerationModeComboBox,
                             BcpdAccelerationMode, self.show_acceleration_mode)
        self.ui.bcpdAccelerationManualGroupBox.setVisible(False)

        self.ui.bcpdGeodesicKernelInputMeshLineEdit.setVisible(False)
        self.ui.bcpdGeodesicKernelInputMeshLabel.setVisible(False)

        self.setup_combo_box(self.ui.bcpdNormalizationComboBox,
                             BcpdNormalizationOptions, None)

        self.ui.bcpdDownsamplingCollapsibleGroupBox.visible = False

        self.ui.bcpdResetParametersPushButton.clicked.connect(
            self.reset_parameters_to_default)
        self.ui.generateModelButton.clicked.connect(self.generate_model)

    def reset_parameters_to_default(self):
        ## Preprocessing parameters ##
        self.ui.preprocessingDownsamplingDistanceThresholdDoubleSpinBox.value = PREPROCESSING_DEFAULT_VALUE_DOWNSAMPLING_DISTANCE_THRESHOLD
        self.ui.preprocessingNormalsEstimationRadiusDoubleSpinBox.value = PREPROCESSING_DEFAULT_VALUE_RADIUS_NORMAL_SCALE
        self.ui.preprocessingNormalsEstimationMaxNeighboursSpinBox.value = PREPROCESSING_DEFAULT_VALUE_MAX_NN_NORMALS
        self.ui.preprocessingFpfhRadiusDoubleSpinBox.value = PREPROCESSING_DEFAULT_VALUE_RADIUS_FEATURE_SCALE
        self.ui.preprocessingFpfhMaxNeighboursSpinBox.value = PREPROCESSING_DEFAULT_VALUE_MAX_NN_FPFH

        ## Registration parameters ##
        self.ui.registrationMaxIterationsSpinBox.value = REGISTRATION_DEFAULT_VALUE_MAX_ITERATIONS
        self.ui.registrationDistanceThresholdDoubleSpinBox.value = REGISTRATION_DEFAULT_VALUE_DISTANCE_THRESHOLD
        self.ui.registrationFitnessThresholdDoubleSpinBox.value = REGISTRATION_DEFAULT_VALUE_FITNESS_THRESHOLD
        self.ui.registrationIcpDistanceThresholdDoubleSpinBox.value = REGISTRATION_DEFAULT_VALUE_ICP_DISTANCE_THRESHOLD

        ## Tuning parameters ##
        self.ui.bcpdOmegaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_OMEGA
        self.ui.bcpdLambdaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_LAMBDA
        self.ui.bcpdBetaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_BETA
        self.ui.bcpdGammaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_GAMMA
        self.ui.bcpdKappaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_KAPPA

        ## Kernel parameters ##
        self.ui.bcpdKernelTypeComboBox.setCurrentIndex(
            BCPD_DEFAULT_VALUE_KERNEL_TYPE)
        self.ui.bcpdStandardKernelComboBox.setCurrentIndex(
            BCPD_DEFAULT_VALUE_STANDARD_KERNEL_TYPE)
        self.ui.bcpdGeodesicKernelTauDoubleSpinBox.value = BCPD_DEFAULT_VALUE_TAU
        self.ui.bcpdGeodesicKernelInputMeshCheckBox.checked = False
        self.ui.bcpdGeodesicKernelInputMeshLineEdit.text = BCPD_DEFAULT_VALUE_INPUT_MESH_PATH
        self.ui.bcpdGeodesicKernelNeighboursSpinBox.value = BCPD_DEFAULT_VALUE_KERNEL_NEIGBOURS
        self.ui.bcpdGeodesicKernelRadiusDoubleSpinBox.value = BCPD_DEFAULT_VALUE_KERNEL_NEIGHBOUR_RADIUS
        self.ui.bcpdGeodesicKernelBetaDoubleSpinBox.value = BCPD_DEFAULT_VALUE_KERNEL_BETA
        self.ui.bcpdGeodesicKernelKTildeDoubleSpinBox.value = BCPD_DEFAULT_VALUE_KERNEL_K_TILDE
        self.ui.bcpdGeodesicKernelEpsilonDoubleSpinBox.value = BCPD_DEFAULT_VALUE_KERNEL_EPSILON

        ## Acceleration parameters ##
        self.ui.bcpdAccelerationModeComboBox.setCurrentIndex(
            BCPD_DEFAULT_VALUE_ACCELERATION_MODE)
        self.ui.bcpdAccelerationAutomaticVbiCheckBox.checked = True
        self.ui.bcpdAccelerationAutomaticPlusPlusCheckBox.checked = True
        self.ui.bcpdAccelerationManualNystormGSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_G
        self.ui.bcpdAccelerationManualNystormJSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_J
        self.ui.bcpdAccelerationManualNystormRSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_R
        self.ui.bcpdAccelerationManualKdTreeScaleDoubleSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_SCALE
        self.ui.bcpdAccelerationManualKdTreeRadiusDoubleSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_RADIUS
        self.ui.bcpdAccelerationManualKdTreeThresholdDoubleSpinBox.value = BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_SIGMA_THRESHOLD

        ## Downsampling options ##
        self.ui.bcpdDownsamplingLineEdit.text = BCPD_DEFAULT_VALUE_DOWNSAMPLING_OPTIONS

        ## Convergence options ##
        self.ui.bcpdConvergenceToleranceDoubleSpinBox.value = BCPD_DEFAULT_VALUE_CONVERGENCE_TOLERANCE
        self.ui.bcpdConvergenceMaxIterationsSpinBox.value = BCPD_DEFAULT_VALUE_CONVERGENCE_MAX_ITERATIONS
        self.ui.bcpdConvergenceMinIterationsSpinBox.value = BCPD_DEFAULT_VALUE_CONVERGENCE_MIN_ITERATIONS

        ## Normalization options ##
        self.ui.bcpdNormalizationComboBox.setCurrentIndex(
            BCPD_DEFAULT_VALUE_NORMALIZATION_OPTIONS)

    def setup_combo_box(self, combo_box: QComboBox, enum, onSelectionChanged):
        """
            Method for setting up combo box and its possible values
        """
        for mode in list(enum):
            combo_box.addItem(mode.name, mode)
        if onSelectionChanged is not None:
            combo_box.currentIndexChanged.connect(onSelectionChanged)
        combo_box.setCurrentIndex(0)

    def show_kernel_type(self, current_index):
        """
            Kernel type callback
        """
        if current_index == BcpdKernelType.STANDARD.value:
            show_standard_setting = True
            show_geodesic_settings = False
        else:
            show_standard_setting = False
            show_geodesic_settings = True

        self.ui.bcpdStandardKernelGroupBox.setVisible(show_standard_setting)
        self.ui.bcpdGeodesicKernelGroupBox.setVisible(show_geodesic_settings)

    def show_acceleration_mode(self, currentIndex):
        """
            Acceleration mode combo box callback
        """
        if currentIndex == BcpdAccelerationMode.AUTOMATIC.value:
            show_automatic = True
            show_manual = False
        else:
            show_automatic = False
            show_manual = True

        self.ui.bcpdAccelerationAutomaticGroupBox.setVisible(show_automatic)
        self.ui.bcpdAccelerationManualGroupBox.setVisible(show_manual)

    def parse_parameters(self) -> dict:
        """
            Parsing parameters from the UI user option elements
        """
        params = {}
        params[PREPROCESSING_KEY] = self.parse_parameters_preprocessing()
        params[BCPD_KEY] = self.parse_parameters_bcpd()

        return params

    def parse_parameters_preprocessing(self) -> dict:
        params = {}

        return params

    def parse_parameters_bcpd(self) -> dict:
        """
            Parsing parameters from the UI for the BCPD stage
        """
        params = {}

        ## Tuning parameters ##
        params[BCPD_VALUE_KEY_OMEGA] = self.ui.bcpdOmegaDoubleSpinBox.value
        params[BCPD_VALUE_KEY_LAMBDA] = self.ui.bcpdLambdaDoubleSpinBox.value
        params[BCPD_VALUE_KEY_BETA] = self.ui.bcpdBetaDoubleSpinBox.value
        params[BCPD_VALUE_KEY_GAMMA] = self.ui.bcpdGammaDoubleSpinBox.value

        kappa = self.ui.bcpdKappaDoubleSpinBox.value
        if kappa < BCPD_MAX_VALUE_KAPPA:  # Setting it to max behaves like "infinity"
            params[BCPD_VALUE_KEY_KAPPA] = kappa

        if self.ui.bcpdAdvancedParametersCheckBox.checked == True:
            self.parse_advanced_parameters(params)

        return params

    def parse_advanced_parameters(self, params: dict) -> None:
        """
            Parsing parameters under the "Advanced" section
        """
        ## Kernel settings ##
        kernel_params = ""
        kernel_type = self.ui.bcpdKernelTypeComboBox.currentIndex
        if (kernel_type == BcpdKernelType.STANDARD.value):
            selected_kernel = self.ui.bcpdStandardKernelComboBox.currentIndex
            # Default kernel is Gauss, which does not need to be specified
            if selected_kernel != BCPD_DEFAULT_VALUE_STANDARD_KERNEL_TYPE:
                kernel_params += str(selected_kernel)
        else:  # Geodesic Kernel
            kernel_params += "geodesic" + BCPD_MULTIPLE_VALUES_SEPARATOR

            tau = self.ui.bcpdGeodesicKernelTauDoubleSpinBox.value
            kernel_params += str(tau) + BCPD_MULTIPLE_VALUES_SEPARATOR

            if self.ui.bcpdGeodesicKernelInputMeshCheckBox.checked == True:
                input_mesh_path = self.ui.bcpdGeodesicKernelInputMeshLineEdit.text
                if not os.path.exists(input_mesh_path):
                    print("File '" + input_mesh_path +
                          "' does not exist. Cancelling process...")
                    return
                kernel_params += input_mesh_path
            else:
                kernel_params += str(self.ui.bcpdGeodesicKernelNeighboursSpinBox.value) + \
                    BCPD_MULTIPLE_VALUES_SEPARATOR
                kernel_params += str(
                    self.ui.bcpdGeodesicKernelRadiusDoubleSpinBox.value)

        if kernel_params != "":
            params[BCPD_VALUE_KEY_KERNEL] = kernel_params

        ## Acceleration settings ##
        if self.ui.bcpdAccelerationModeComboBox.currentIndex == BcpdAccelerationMode.AUTOMATIC.value:
            if self.ui.bcpdAccelerationAutomaticVbiCheckBox.checked == True:
                params[BCPD_VALUE_KEY_NYSTORM_G] = 70
                params[BCPD_VALUE_KEY_NYSTORM_P] = 300
                # Option switch without a value
                params[BCPD_VALUE_KEY_KD_TREE] = ""
                params[BCPD_VALUE_KEY_KD_TREE_SCALE] = 7
                params[BCPD_VALUE_KEY_KD_TREE_RADIUS] = 0.15
            if self.ui.bcpdAccelerationAutomaticPlusPlusCheckBox.checked == True:
                params[BCPD_VALUE_KEY_DOWNSAMPLING] = "B,10000,0.08"
        else:  # Manual acceleration
            if self.ui.bcpdAccelerationManualNystormGroupBox.checked == True:
                params[
                    BCPD_VALUE_KEY_NYSTORM_G] = self.ui.bcpdAccelerationManualNystormGSpinBox.value
                params[
                    BCPD_VALUE_KEY_NYSTORM_P] = self.ui.bcpdAccelerationManualNystormJSpinBox.value
                params[
                    BCPD_VALUE_KEY_NYSTORM_R] = self.ui.bcpdAccelerationManualNystormRSpinBox.value
            if self.ui.bcpdAccelerationManualKdTreeGroupBox.checked == True:
                # Option switch without a value
                params[BCPD_VALUE_KEY_KD_TREE] = ""
                params[
                    BCPD_VALUE_KEY_KD_TREE_SCALE] = self.ui.bcpdAccelerationManualKdTreeScaleDoubleSpinBox.value
                params[
                    BCPD_VALUE_KEY_KD_TREE_RADIUS] = self.ui.bcpdAccelerationManualKdTreeRadiusDoubleSpinBox.value
                params[
                    BCPD_VALUE_KEY_KD_TREE_THRESHOLD] = self.ui.bcpdAccelerationManualKdTreeThresholdDoubleSpinBox.value

        ## Downsampling settings ##

        if params.get(BCPD_VALUE_KEY_DOWNSAMPLING) is None:
            params[
                BCPD_VALUE_KEY_DOWNSAMPLING] = self.ui.bcpdDownsamplingLineEdit.text

        ## Convergence options ##
        params[
            BCPD_VALUE_KEY_CONVERGENCE_TOLERANCE] = self.ui.bcpdConvergenceToleranceDoubleSpinBox.value
        params[
            BCPD_VALUE_KEY_CONVERGENCE_MIN_ITERATIONS] = self.ui.bcpdConvergenceMinIterationsSpinBox.value
        params[
            BCPD_VALUE_KEY_CONVERGENCE_MAX_ITERATIONS] = self.ui.bcpdConvergenceMaxIterationsSpinBox.value

        ## Normalization options ##
        params[
            BCPD_VALUE_KEY_NORMALIZATION] = self.ui.bcpdNormalizationComboBox.currentText.lower(
            )

    def generate_model(self) -> None:
        """
            Generate button callback. Calls the Logic's generate_model method and adds the results into the scene
        """
        params = self.parse_parameters()
        params_string = ""
        for key in params.keys():
            params_string += key + str(params[key]) + "\n"
        print("Parameters: " + params_string)

        # err, generated_polydata, merged_polydata = self.logic.generate_model(self.ui.sourceNodeSelectionBox.currentNode(), self.ui.targetNodeSelectionBox.currentNode(), params_string)

        # if (err == EXIT_OK):
        #     model_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'BCPD generated')
        #     model_node.SetAndObservePolyData(generated_polydata)
        #     model_node.CreateDefaultDisplayNodes()

        #     model_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'BCPD merged')
        #     model_node.SetAndObservePolyData(merged_polydata)
        #     model_node.CreateDefaultDisplayNodes()
