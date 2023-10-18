from slicer.ScriptedLoadableModule import ScriptedLoadableModule
from src.widget.SlicerBoneMorphingWidget import SlicerBoneMorphingWidget

class SlicerBoneMorphing(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Slicer Bone Mesh Morphing Module" 
    self.parent.categories = ["Examples"] # TODO: Change to an appropriate category
    self.parent.dependencies = []
    self.parent.contributors = ["Jan Heres (University of West Bohemia), Eva C. Herbst (ETH Zurich), Arthur Porto (Louisiana State University)"] 
    self.parent.helpText = """
        This module gives a user ability to recreate bone mesh models based on their partial scan.
        Please, start with importing your model in checking out the options on the left side.  
    """
    self.parent.acknowledgementText = """
        Credits: Jan Heres
        I would love to thank my awesome colleagues Eva C. Herbst and Arthur Porto for contributing to this project! 
    """    
