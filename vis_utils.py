import numpy as np
import vtk
from vtk.util import numpy_support
from scipy.ndimage import zoom


def load_volume(file, scale=None):
    raw_arr = np.fromfile(file, dtype=np.uint8)
    volume_dims_str = file.split('_')[-1].split('.')[0].split('x')

    dims_order = (2, 1, 0)
    volume_dims = [int(volume_dims_str[i]) for i in dims_order]
    arr = raw_arr.reshape(volume_dims)

    if scale is not None:
        arr = zoom(arr, scale, order=0)

    return arr


def get_transform_function(mid_point, delta, central_value, margin_value):
    transform_function = vtk.vtkPiecewiseFunction()
    transform_function.AddPoint(mid_point - delta, margin_value)
    transform_function.AddPoint(mid_point, central_value)
    transform_function.AddPoint(mid_point + delta, margin_value)
    return transform_function

def get_color_function(mid_point, primary_color):
    volumeColor = vtk.vtkColorTransferFunction()
    volumeColor.AddRGBPoint(mid_point - 50, *(0, 0, 0))
    volumeColor.AddRGBPoint(mid_point, *primary_color)
    volumeColor.AddRGBPoint(mid_point + 50, *(0, 0, 0))
    return volumeColor

class SliderWrapper:
    def __init__(self, title_text, value_range, initial_value, position):
        slider_rep = vtk.vtkSliderRepresentation2D()
        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        slider_rep.GetPoint1Coordinate().SetValue(*position[0])
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        slider_rep.GetPoint2Coordinate().SetValue(*position[1])
        slider_rep.SetMinimumValue(value_range[0])
        slider_rep.SetMaximumValue(value_range[1])
        slider_rep.SetValue(initial_value)
        slider_rep.SetTitleText(title_text)

        self._slider_rep = slider_rep

    def get_widget(self, interactor):
        widget = vtk.vtkSliderWidget()
        widget.SetInteractor(interactor)
        widget.SetRepresentation(self._slider_rep)
        widget.SetAnimationModeToAnimate()
        widget.EnabledOn()

        return widget


class VolumeVisualizer:
    def __init__(self, volume, data_scalar_range='auto'):
        self._volume = volume

        if data_scalar_range == 'auto':
            data_scalar_range = (int(volume.min()), int(volume.max() + 1))

        self._data_scalar_range = data_scalar_range

        self._dynamic_properties = {
            'opacity_function_midpoint': (data_scalar_range[0] + data_scalar_range[1]) / 2.,
            'opacity_function_max': 1.,
            'opacity_function_width': (data_scalar_range[1] - data_scalar_range[0]) / 2.
        }

    def visualize(self, scale=1., interpolation_order=0, primary_color=None):
        volume = zoom(self._volume, scale, order=interpolation_order)

        flat_volume = volume.transpose((2, 1, 0)).flatten()

        # --- data_importer
        data_importer = vtk.vtkImageImport()
        data_string = flat_volume.tostring()
        data_importer.CopyImportVoidPointer(data_string, len(data_string))
        data_importer.SetDataScalarTypeToUnsignedChar()
        data_importer.SetNumberOfScalarComponents(1)
        data_importer.SetDataExtent(0, volume.shape[0] - 1, 0, volume.shape[1] - 1, 0, volume.shape[2] - 1)
        data_importer.SetWholeExtent(0, volume.shape[0] - 1, 0, volume.shape[1] - 1, 0, volume.shape[2] - 1)

        # --- mapper
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputConnection(data_importer.GetOutputPort())

        # --- actor
        actor = vtk.vtkVolume()
        actor.SetMapper(mapper)
        if primary_color is not None:
            actor.GetProperty().SetColor(0, get_color_function(self._dynamic_properties['opacity_function_midpoint'], primary_color))

        # --- renderer
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)

        # --- window
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 600)

        # --- interactor
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)

        def slider_callback_wrapper(property_name):
            def callback(caller, ev):
                value = caller.GetSliderRepresentation().GetValue()
                self._dynamic_properties[property_name] = value
                transform_function = get_transform_function(
                    mid_point=self._dynamic_properties['opacity_function_midpoint'],
                    delta=self._dynamic_properties['opacity_function_width'] / 2.,
                    margin_value=0.,
                    central_value=self._dynamic_properties['opacity_function_max']
                )
                actor.GetProperty().SetScalarOpacity(0, transform_function)
                if primary_color is not None:
                    actor.GetProperty().SetColor(0, get_color_function(self._dynamic_properties['opacity_function_midpoint'], primary_color))
                render_window.Render()

            return callback

        midpoint_slider_widget = SliderWrapper(
            title_text='opacity function midpoint',
            value_range=self._data_scalar_range,
            initial_value=self._dynamic_properties['opacity_function_midpoint'],
            position=((.7, .1), (.9, .1))
        ).get_widget(interactor)
        midpoint_slider_widget.AddObserver('InteractionEvent', slider_callback_wrapper('opacity_function_midpoint'))

        width_slider_widget = SliderWrapper(
            title_text='opacity function width',
            value_range=(0, self._data_scalar_range[1] - self._data_scalar_range[0]),
            initial_value=self._dynamic_properties['opacity_function_width'],
            position=((.7, .25), (.9, .25))
        ).get_widget(interactor)
        width_slider_widget.AddObserver('InteractionEvent', slider_callback_wrapper('opacity_function_width'))

        opacity_slider_widget = SliderWrapper(
            title_text='max opacity',
            value_range=(0., 1.),
            initial_value=1.,
            position=((.7, .4), (.9, .4))
        ).get_widget(interactor)
        opacity_slider_widget.AddObserver('InteractionEvent', slider_callback_wrapper('opacity_function_max'))

        # --- start
        transform_function = get_transform_function(
            mid_point=self._dynamic_properties['opacity_function_midpoint'],
            delta=self._dynamic_properties['opacity_function_width'] / 2.,
            margin_value=0.,
            central_value=self._dynamic_properties['opacity_function_max']
        )
        actor.GetProperty().SetScalarOpacity(0, transform_function)
        render_window.Render()
        style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)
        interactor.Initialize()
        interactor.Start()