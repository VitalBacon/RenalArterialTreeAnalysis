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


def visualize_volume(np_volume, scale=1, dims_order=(2, 1, 0), data_scalar_range=(0, 255),
                     starting_mid_point=100, transform_function_delta=50,
                     primary_color=None, background_color=(0, 0, 0), color_function_delta=20):
    def get_transform_function(mid_point):
        transform_function = vtk.vtkPiecewiseFunction()
        transform_function.AddPoint(mid_point - transform_function_delta, 0)
        transform_function.AddPoint(mid_point, 1)
        transform_function.AddPoint(mid_point + transform_function_delta, 0)
        return transform_function

    def get_color_function(mid_point):
        volumeColor = vtk.vtkColorTransferFunction()
        volumeColor.AddRGBPoint(mid_point - color_function_delta, *background_color)
        volumeColor.AddRGBPoint(mid_point, *primary_color)
        volumeColor.AddRGBPoint(mid_point + color_function_delta, *background_color)
        return volumeColor

    np_volume = zoom(np_volume, scale, order=1)
    flat_volume = np_volume.transpose(dims_order).flatten()

    # --- dataImporter
    dataImporter = vtk.vtkImageImport()
    data_string = flat_volume.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.SetDataExtent(0, np_volume.shape[0] - 1, 0, np_volume.shape[1] - 1, 0, np_volume.shape[2] - 1)
    dataImporter.SetWholeExtent(0, np_volume.shape[0] - 1, 0, np_volume.shape[1] - 1, 0, np_volume.shape[2] - 1)

    # --- mapper
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputConnection(dataImporter.GetOutputPort())

    # --- actor
    actor = vtk.vtkVolume()
    actor.SetMapper(mapper)
    actor.GetProperty().SetScalarOpacity(0, get_transform_function(starting_mid_point))
    if primary_color is not None:
        actor.GetProperty().SetColor(0, get_color_function(starting_mid_point))

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
        def __init__(self, volume_actor, render_window):
            self.render_window = render_window
            self.volume_actor = volume_actor

        def __call__(self, caller, ev):
            value = int(caller.GetSliderRepresentation().GetValue())
            self.volume_actor.GetProperty().SetScalarOpacity(0, get_transform_function(value))
            if primary_color is not None:
                self.volume_actor.GetProperty().SetColor(0, get_color_function(value))
            self.render_window.Render()

    sliderRep = vtk.vtkSliderRepresentation2D()
    sliderRep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep.GetPoint1Coordinate().SetValue(.7, .1)
    sliderRep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep.GetPoint2Coordinate().SetValue(.9, .1)
    sliderRep.SetMinimumValue(data_scalar_range[0] + 1)
    sliderRep.SetMaximumValue(data_scalar_range[1])
    sliderRep.SetValue(starting_mid_point)
    sliderRep.SetTitleText("Transfer function midpoint")

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
