import src.logic.Constants 
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
      self.ui.bcpdOmegaDoubleSpinBox.value = 0.1

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

    


            
        
