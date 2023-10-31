import src.logic.Constants as Constants
import ctk
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleWidget
from src.logic.SlicerBoneMorphingLogic import SlicerBoneMorphingLogic

from src.logic.Enums import BCPDAccelerationMode, BCPDKernelMode, BCPDNormalizationOptions, BCPDStandardKernelMode
from enum import Enum
from qt import QComboBox

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
    self.uiWidget = slicer.util.loadUI(self.resourcePath("UI/SlicerBoneMorphing.ui"))
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

    self.setup_combo_box(self.ui.bcpdKernelTypeComboBox, BCPDKernelMode, self.show_kernel_type)
    self.ui.bcpdGeodesicKernelGroupBox.setVisible(False)

    self.setup_combo_box(self.ui.bcpdStandardKernelComboBox, BCPDStandardKernelMode, None)

    self.setup_combo_box(self.ui.bcpdAccelerationModeComboBox, BCPDAccelerationMode, self.show_acceleration_mode)
    self.ui.bcpdAccelerationManualGroupBox.setVisible(False)

    self.ui.bcpdGeodesicKernelInputMeshLineEdit.setVisible(False)
    self.ui.bcpdGeodesicKernelInputMeshLabel.setVisible(False)

    self.setup_combo_box(self.ui.bcpdNormalizationComboBox, BCPDNormalizationOptions, None)
 
    self.ui.bcpdResetParametersPushButton.clicked.connect(self.reset_parameters_to_default)
    self.ui.generateModelButton.clicked.connect(self.generate_model)


  def reset_parameters_to_default(self):
      ## Tuning parameters ## 
      self.ui.bcpdOmegaDoubleSpinBox.value = Constants.BCPD_DEFAULT_VALUE_OMEGA
      self.ui.bcpdLambdaDoubleSpinBox.value = Constants.BCPD_DEFAULT_VALUE_LAMBDA
      self.ui.bcpdBetaDoubleSpinBox.value = Constants.BCPD_DEFAULT_VALUE_BETA
      self.ui.bcpdGammaDoubleSpinBox.value = Constants.BCPD_DEFAULT_VALUE_GAMMA
      self.ui.bcpdKappaDoubleSpinBox.value = Constants.BCPD_DEFAULT_VALUE_KAPPA

      # self.ui.bcpdAdvancedParametersCheckBox.checked = False
      
      ## Kernel parameters ##
      self.ui.bcpdKernelTypeComboBox.setCurrentIndex(Constants.BCPD_DEFAULT_VALUE_KERNEL_TYPE)
      self.ui.bcpdStandardKernelComboBox.setCurrentIndex(Constants.BCPD_DEFAULT_VALUE_STANDARD_KERNEL_TYPE)
      self.ui.bcpdGeodesicKernelTauDoubleSpinBox.value = Constants.BCPD_DEFAULT_VALUE_TAU
      self.ui.bcpdGeodesicKernelInputMeshCheckBox.checked = False
      self.ui.bcpdGeodesicKernelInputMeshLineEdit.text = Constants.BCPD_DEFAULT_VALUE_INPUT_MESH_PATH
      self.ui.bcpdGeodesicKernelNeighboursSpinBox.value = Constants.BCPD_DEFAULT_VALUE_KERNEL_NEIGBOURS
      self.ui.bcpdGeodesicKernelRadiusDoubleSpinBox.value = Constants.BCPD_DEFAULT_VALUE_KERNEL_NEIGHBOUR_RADIUS
      self.ui.bcpdGeodesicKernelBetaDoubleSpinBox.value = Constants.BCPD_DEFAULT_VALUE_KERNEL_BETA
      self.ui.bcpdGeodesicKernelKTildeDoubleSpinBox.value = Constants.BCPD_DEFAULT_VALUE_KERNEL_K_TILDE
      self.ui.bcpdGeodesicKernelEpsilonDoubleSpinBox.value = Constants.BCPD_DEFAULT_VALUE_KERNEL_EPSILON


      ## Acceleration parameters ##
      self.ui.bcpdAccelerationModeComboBox.setCurrentIndex(Constants.BCPD_DEFAULT_VALUE_ACCELERATION_MODE)
      self.ui.bcpdAccelerationAutomaticVbiCheckBox.checked = True
      self.ui.bcpdAccelerationAutomaticPlusPlusCheckBox.checked = True
      self.ui.bcpdAccelerationManualNystormGSpinBox.value = Constants.BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_G
      self.ui.bcpdAccelerationManualNystormJSpinBox.value = Constants.BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_J
      self.ui.bcpdAccelerationManualNystormRSpinBox.value = Constants.BCPD_DEFAULT_VALUE_ACCELERATION_NYSTORM_SAMPLES_R
      self.ui.bcpdAccelerationManualKdTreeScaleDoubleSpinBox.value = Constants.BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_SCALE
      self.ui.bcpdAccelerationManualKdTreeRadiusDoubleSpinBox.value = Constants.BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_RADIUS
      self.ui.bcpdAccelerationManualKdTreeThresholdDoubleSpinBox.value = Constants.BCPD_DEFAULT_VALUE_ACCELERATION_KD_TREE_SIGMA_THRESHOLD

      ## Downsampling options ##
      self.ui.bcpdDownsamplingLineEdit.text = Constants.BCPD_DEFAULT_VALUE_DOWNSAMPLING_OPTIONS

      ## Convergence options ## 
      self.ui.bcpdConvergenceToleranceDoubleSpinBox.value = Constants.BCPD_DEFAULT_VALUE_CONVERGENCE_TOLERANCE
      self.ui.bcpdConvergenceMaxIterationsSpinBox.value = Constants.BCPD_DEFAULT_VALUE_CONVERGENCE_MAX_ITERATIONS
      self.ui.bcpdConvergenceMinIterationsSpinBox.value = Constants.BCPD_DEFAULT_VALUE_CONVERGENCE_MIN_ITERATIONS

      ## Normalization options ## 
      self.ui.bcpdNormalizationComboBox.setCurrentIndex(Constants.BCPD_DEFAULT_VALUE_NORMALIZATION_OPTIONS)
      


  def setup_combo_box(self, comboBox: QComboBox, enum, onSelectionChanged): 
    """
        Method for setting up combo box and its possible values
    """
    for mode in list(enum):
        comboBox.addItem(mode.name, mode)
    if(onSelectionChanged != None):
        comboBox.currentIndexChanged.connect(onSelectionChanged)
    comboBox.setCurrentIndex(0) 

  def show_kernel_type(self, selectedIndex):
    """
        Kernel type callback
    """
    if selectedIndex == BCPDKernelMode.STANDARD.value: 
        showStandardSettings = True
        showGeodesicSettings = False
    else:
        showStandardSettings = False
        showGeodesicSettings = True

    self.ui.bcpdStandardKernelGroupBox.setVisible(showStandardSettings)
    self.ui.bcpdGeodesicKernelGroupBox.setVisible(showGeodesicSettings)

  def show_acceleration_mode(self, selectedIndex):
    """
        Acceleration mode combo box callback
    """
    if selectedIndex == BCPDAccelerationMode.AUTOMATIC.value:
        showAutomatic = True 
        showManual = False
    else:
        showAutomatic = False
        showManual = True
    
    self.ui.bcpdAccelerationAutomaticGroupBox.setVisible(showAutomatic)
    self.ui.bcpdAccelerationManualGroupBox.setVisible(showManual)

    def parse_parameters(self): 
        params = [] 

        return params


  def generate_model(self):
    """
        Generate button callback
    """
    # mode = self.ui.bcpdModeComboBox.currentData
    # if(mode == None):
    #     print("Please, select a proper BCPD registration mode")
    # self.bcpd_options["mode"] = mode

    dvl = self.ui.bcpdDeformationVectorLengthSpinBox.value
    self.bcpd_options["b"] = dvl


      # code, vtk_polydata = self.logic.generate_model(self.ui.sourceNodeSelectionBox.currentNode(), self.ui.targetNodeSelectionBox.currentNode())

      # if(code == EXIT_OK):
      #   model_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'BCPD result')
      #   model_node.SetAndObservePolyData(vtk_polydata)
      #   model_node.CreateDefaultDisplayNodes()  

    


            
        
