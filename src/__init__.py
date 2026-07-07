try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from .resview_widget import ResviewDockWidget

__all__ = ("ResviewDockWidget",)
