import numpy as np
import vtk
from vtk.util import numpy_support
from scipy.ndimage import zoom


def load_volume(file, dims_order=(2, 1, 0), scale=None):
    raw_arr = np.fromfile(file, dtype=np.uint8)
    volume_dims_str = file.split('_')[-1].split('.')[0].split('x')
    volume_dims = [int(volume_dims_str[i]) for i in dims_order]
    arr = raw_arr.reshape(volume_dims)
    
    if scale is not None:
        arr = zoom(arr, scale, order=0)
        
    return arr


def visualize_volume(np_volume, scale=1, t=(2, 1, 0)):
    
    np_volume = zoom(np_volume, scale, order=1)
    flat_volume = np_volume.transpose(t).flatten()

    def get_transform_function(mid_point, delta=50):
        transform_function = vtk.vtkPiecewiseFunction()
        transform_function.AddPoint(mid_point - delta, 0)
        transform_function.AddPoint(mid_point, 1)
        transform_function.AddPoint(mid_point + delta, 0)
        return transform_function
    
    def get_color_function(mid_point, delta, primary_color=(0, 100, 0), background_color=(0, 0, 0)):
        volumeColor = vtk.vtkColorTransferFunction()
        volumeColor.AddRGBPoint(mid_point - delta, *background_color)
        volumeColor.AddRGBPoint(mid_point, *primary_color)
        volumeColor.AddRGBPoint(mid_point + delta, *background_color)
        return volumeColor

#     # --- source: read data
#     dir = 'mr_brainixA'
#     reader = vtk.vtkArrayDataReader()
#     reader.SetDirectoryName(dir)
#     reader.Update()
#     imageData = reader.GetOutput()
    
    
    dataImporter = vtk.vtkImageImport()
    data_string = flat_volume.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    
    dataImporter.SetDataExtent(0, np_volume.shape[0]-1, 0, np_volume.shape[1]-1, 0, np_volume.shape[2]-1)
    dataImporter.SetWholeExtent(0, np_volume.shape[0]-1, 0, np_volume.shape[1]-1, 0, np_volume.shape[2]-1)

    dataScalarRange = (0, 255) #imageData.GetScalarRange()
    # print(imageData.GetDimensions())
    # print(imageData)

    nFrames = np_volume.shape[2] #imageData.GetDimensions()[2]

    # mapper
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputConnection(dataImporter.GetOutputPort())
#     mapper.SetInputConnection(reader.GetOutputPort())

    # actor
    actor = vtk.vtkVolume()
    actor.SetMapper(mapper)
    actor.GetProperty().SetScalarOpacity(0, get_transform_function(65, delta=50))
    actor.GetProperty().SetColor(0, get_color_function(65, delta=50))

    # --- renderer
    ren1 = vtk.vtkRenderer()
    ren1.AddActor(actor)

    # --- window
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren1)
    renWin.SetSize(800, 600)

    # --- interactor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # --- slider to change frame: callback class, sliderRepresentation, slider
    class FrameCallback(object):
        def __init__(self, actor, renWin):
            self.renWin = renWin
            self.actor = actor
        def __call__(self, caller, ev):
            value = int(caller.GetSliderRepresentation().GetValue())
            
            self.actor.GetProperty().SetColor(0, get_color_function(value, delta=50))
            self.actor.GetProperty().SetScalarOpacity(0, get_transform_function(value, delta=50))
            self.renWin.Render()

    sliderRep = vtk.vtkSliderRepresentation2D()
    sliderRep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep.GetPoint1Coordinate().SetValue(.7, .1)
    sliderRep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep.GetPoint2Coordinate().SetValue(.9, .1)
    sliderRep.SetMinimumValue(dataScalarRange[0] + 1)
    sliderRep.SetMaximumValue(dataScalarRange[1])
    sliderRep.SetValue(1)
    sliderRep.SetTitleText("transfer function midpoint")

    slider = vtk.vtkSliderWidget()
    slider.SetInteractor(iren)
    slider.SetRepresentation(sliderRep)
    slider.SetAnimationModeToAnimate()
    slider.EnabledOn()
    slider.AddObserver('InteractionEvent', FrameCallback(actor, renWin))

    # --- run
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.Initialize()
    iren.Start()