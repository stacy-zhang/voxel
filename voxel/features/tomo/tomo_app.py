import os
import asyncio
import base64
import gc
import io
import math
import time
from pathlib import Path
import pathlib
from typing import Optional, Tuple
import importlib
import sys
import types
import argparse

import numpy as np

from trame.app import get_server # Trame server framework
from trame.ui.vuetify3 import SinglePageWithDrawerLayout 
from trame.ui.vuetify3 import SinglePageLayout 
from trame.widgets import vuetify3 as v3, vtk as vtk_widgets, html

from vtkmodules.vtkCommonDataModel import (
    vtkPlane, # for slicing the volume with axis-aligned planes
    vtkPiecewiseFunction, # for defining the opacity transfer function (mapping scalar values to opacity)
)

from vtkmodules.vtkFiltersModeling import vtkOutlineFilter # to create a wireframe box around the volume
from vtkmodules.vtkIOImage import vtkTIFFReader # to read 3D TIFF files as vtkImageData
from vtkmodules.vtkRenderingCore import (
    vtkActor, # for rendering the outline box
    vtkColorTransferFunction, # for defining the color transfer function (mapping scalar values to colors)
    vtkPolyDataMapper, # to map the outline geometry to graphics primitives
    vtkRenderer, # the main rendering engine that manages the scene
    vtkRenderWindow, # the window that displays the rendered scene
    vtkRenderWindowInteractor, # handles user interaction (mouse, keyboard) with the render window
    vtkVolume, # the actor type for volume rendering
    vtkVolumeProperty, # holds the properties of the volume rendering (color, opacity, shading, etc.)
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper # the GPU-accelerated volume mapper that does the actual rendering of the 3D data

from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera # allows the user to rotate/zoom/pan the view with mouse interactions (trackball style)
import vtkmodules.vtkInteractionStyle  # noqa – required
import vtkmodules.vtkRenderingOpenGL2  # noqa – required
# the above two imports are needed to ensure the appropriate VTK rendering and interaction styles are registered, even if we don't directly reference them in the code


def create_tomo_server():
    ################
    # VTK pipeline #
    ################
    renderer = vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.1)  # dark gray background

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1024, 768) # initial window size (can be resized by user)
    render_window.SetOffScreenRendering(1) # enable off-screen rendering for web-based applications (no native window needed)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
    interactor.EnableRenderOff()

    volume_mapper = vtkSmartVolumeMapper()
    volume_mapper.SetBlendModeToComposite() # default blend mode (user can change to max/min/avg intensity)

    color_tf = vtkColorTransferFunction()
    opacity_tf = vtkPiecewiseFunction()

    vol_property = vtkVolumeProperty()
    vol_property.SetIndependentComponents(True)
    vol_property.SetInterpolationTypeToLinear() # smooth interpolation between voxels for better quality (can be set to nearest for sharper but blockier look)
    vol_property.SetColor(color_tf)
    vol_property.SetScalarOpacity(opacity_tf)
    vol_property.ShadeOn() # enable lighting by default for better depth perception, user can toggle off if desired
    vol_property.SetAmbient(0.2)
    vol_property.SetDiffuse(0.7)
    vol_property.SetSpecular(0.3)
    vol_property.SetSpecularPower(10.0) # shininess of the specular highlight (higher = smaller, sharper highlight)

    volume_actor = vtkVolume()
    volume_actor.SetMapper(volume_mapper)
    volume_actor.SetProperty(vol_property)
    volume_actor.VisibilityOff() # start with volume hidden until a file is loaded
    renderer.AddVolume(volume_actor)

    renderer.ResetCamera() # reset camera to fit the scene

    ##########################
    # Trame server and state #
    ##########################
    server = get_server(name="tomo_app")
    state, ctrl = server.state, server.controller

    ###########
    # helpers #
    ###########

    ##########################
    # @state.change handlers #
    ##########################

    ######
    # UI #
    ######
    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("Tomography Viewer")
        with layout.drawer as drawer:
            drawer.width = 360

            with v3.VCard(flat=True, classes="mt-5 pb-4"):
                v3.VCardTitle("Pipeline", classes="text-h6 font-weight-regular")
            v3.VDivider()
            with v3.VCard(flat=True, classes="mt-4 pb-4"):
                v3.VCardTitle("Properties", classes="text-h6 font-weight-regular")
            v3.VDivider()

        with layout.content:
            with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                view = vtk_widgets.VtkRemoteView(
                    render_window,
                    ref="view",
                    interactive_ratio=0.5,
                    still_ratio=1,
                )
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera

    return server

def run_server(port=0, host="localhost", open_browser=True):
    create_tomo_server().start(port=port, host=host, open_browser=open_browser)

def main(argv=None):
    # CLI
    parser = argparse.ArgumentParser(description="Tomography Web Application using Trame")
    parser.add_argument("--port", type=int, default=0, help="Port to bind the Trame server (0 = auto)")
    parser.add_argument("--host", type=str, default="localhost", help="Host to run the server on (default: localhost)")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    run_server(port=args.port, host=args.host, open_browser=not args.no_browser)