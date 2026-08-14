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


def create_tomo_server():
    server = get_server(name="tomo_app")
    
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