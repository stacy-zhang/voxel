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
from trame.widgets import vuetify3 as vuetify, vtk as vtk_widgets, html

# CLI
parser = argparse.ArgumentParser(description="Tomography Web Application using Trame")
parser.add_argument("--port", type=int, default=0, help="Port to bind the Trame server (0 = auto)")
parser.add_argument("--host", type=str, default="localhost", help="Host to run the server on (default: localhost)")
parser.add_argument("--open-browser", action="store_true", help="Open the browser on server start")


def create_tomo_server():
    server = get_server(name="tomo_app")
    
    return server


if __name__ == "__main__":
    args = parser.parse_args()
    server = create_tomo_server()
    server.start(port=args.port, host=args.host, open_browser=args.open_browser)