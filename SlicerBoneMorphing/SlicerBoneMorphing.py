from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from src.main import *

class SlicerBoneMorphing(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Slicer Bone Mesh Module" 
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["Jan Heres"] 
    self.parent.helpText = """
        This is a testing module help text. 
    """
    self.parent.acknowledgementText = """
        Credits: Jan Heres
    """    
