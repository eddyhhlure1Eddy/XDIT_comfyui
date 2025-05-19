import os
import sys
import torch
from pathlib import Path

# Import compatibility layer
from .compat import setup_xdit_path, setup_env_vars

# Setup environment
setup_xdit_path()
setup_env_vars()

# Import custom nodes
from .nodes.loader import XDiTUNETLoader, XfuserPipelineLoader
from .nodes.sampler import XDiTSamplerCustomAdvanced, XfuserSampler
from .nodes.clip import XfuserClipTextEncode

# Version information
__version__ = "0.1.0"

# Register nodes
NODE_CLASS_MAPPINGS = {
    "XDiTUNETLoader": XDiTUNETLoader,
    "XDiTSamplerCustomAdvanced": XDiTSamplerCustomAdvanced,
    "XfuserPipelineLoader": XfuserPipelineLoader,
    "XfuserClipTextEncode": XfuserClipTextEncode,
    "XfuserSampler": XfuserSampler,
}

# Set node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "XDiTUNETLoader": "xDiT UNET Loader",
    "XDiTSamplerCustomAdvanced": "xDiT Sampler Advanced",
    "XfuserPipelineLoader": "Xfuser Pipeline Loader",
    "XfuserClipTextEncode": "Xfuser CLIP Text Encode",
    "XfuserSampler": "Xfuser Sampler",
}

print("xDiT Nodes for ComfyUI loaded successfully!") 