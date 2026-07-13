"""Static UI assets and constants for the browser control panel.

These are presentation-layer values with no behavior: the colormap names shown
in the dropdowns, the inline SVG glyphs for the layer visibility toggles, and
the default frame count used to seed the scan-range inputs. Keeping them here
lets the UI be restyled or extended without touching the pipeline or render
code.
"""

# Colormap choices offered in the View-tab / layer-panel colormap dropdowns.
COLORMAP_NAMES = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "coolwarm",
    "gray",
]

# Default number of frames to pre-select when a TIFF directory is chosen. The
# scan-range inputs are seeded so the first DEFAULT_FRAME_COUNT frames load.
DEFAULT_FRAME_COUNT = 362

# Layer-panel eye toggle glyphs, rendered as inline SVG via Vue's v-html (see
# the layer panel). Using inline SVG instead of a CSS url() data URI avoids the
# percent-encoding pitfalls that left the icon invisible, and needs no icon font
# or Vuetify component. These are the Material Design Icons "eye" / "eye-off"
# paths on a 24x24 viewBox.
_EYE_ON_SVG = (
    "<svg viewBox='0 0 24 24' width='20' height='20' style='display:block;'>"
    "<path fill='#dcdce0' d='M12,9A3,3 0 0,0 9,12A3,3 0 0,0 12,15A3,3 0 0,0 "
    "15,12A3,3 0 0,0 12,9M12,17A5,5 0 0,1 7,12A5,5 0 0,1 12,7A5,5 0 0,1 17,12A5,5 "
    "0 0,1 12,17M12,4.5C7,4.5 2.73,7.61 1,12C2.73,16.39 7,19.5 12,19.5C17,19.5 "
    "21.27,16.39 23,12C21.27,7.61 17,4.5 12,4.5Z'/></svg>"
)
_EYE_OFF_SVG = (
    "<svg viewBox='0 0 24 24' width='20' height='20' style='display:block;'>"
    "<path fill='#808088' d='M11.83,9L15,12.16C15,12.11 15,12.05 15,12A3,3 0 0,0 "
    "12,9C11.94,9 11.89,9 11.83,9M7.53,9.8L9.08,11.35C9.03,11.56 9,11.77 9,12A3,3 "
    "0 0,0 12,15C12.22,15 12.44,14.97 12.65,14.92L14.2,16.47C13.53,16.8 12.79,17 "
    "12,17A5,5 0 0,1 7,12C7,11.21 7.2,10.47 7.53,9.8M2,4.27L4.28,6.55L4.73,7C3.08,"
    "8.3 1.78,10 1,12C2.73,16.39 7,19.5 12,19.5C13.55,19.5 15.03,19.2 "
    "16.38,18.66L16.81,19.09L19.73,22L21,20.73L3.27,3M12,7A5,5 0 0,1 17,12C17,12.64 "
    "16.87,13.26 16.64,13.82L19.57,16.75C21.07,15.5 22.27,13.86 23,12C21.27,7.61 "
    "17,4.5 12,4.5C10.6,4.5 9.26,4.75 8,5.2L10.17,7.35C10.74,7.13 11.35,7 12,7Z'/>"
    "</svg>"
)
