import warnings
from enum import Enum, unique
warnings.filterwarnings('ignore')
import os
import torch
import logging
import platform
import stat
# Replaced problematic fsplit import with custom implementation
from tools.file_utils import Filesplit

# Project version number
VERSION = "1.1.1"
# ×××××××××××××××××××× [Do not change] start ××××××××××××××××××××
logging.disable(logging.DEBUG)  # Turn off DEBUG log printing
logging.disable(logging.WARNING)  # Turn off WARNING log printing
try:
    import torch_directml
    device = torch_directml.device(torch_directml.default_device())
    USE_DML = True
except:
    USE_DML = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Try to import onnxruntime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

# Whether to use ONNX (DirectML/AMD/Intel)
ONNX_PROVIDERS = []
if ONNX_AVAILABLE:
    available_providers = ort.get_available_providers()
    for provider in available_providers:
        if provider in [
            "CPUExecutionProvider"
        ]:
            continue
        if provider not in [
            "DmlExecutionProvider",         # DirectML, suitable for Windows GPU
            "ROCMExecutionProvider",        # AMD ROCm
            "MIGraphXExecutionProvider",    # AMD MIGraphX
            "VitisAIExecutionProvider",     # AMD VitisAI, suitable for RyzenAI & Windows, actual performance seems similar to DirectML
            "OpenVINOExecutionProvider",    # Intel GPU
            "MetalExecutionProvider",       # Apple macOS
            "CoreMLExecutionProvider",      # Apple macOS
            "CUDAExecutionProvider",        # Nvidia GPU
        ]:
            continue
        ONNX_PROVIDERS.append(provider)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAMA_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'big-lama')
STTN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn', 'infer_model.pth')
VIDEO_INPAINT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'video')
MODEL_VERSION = 'V4'
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')

# Check if there is a complete model file under this path, if not, merge small files to generate a complete file
if 'big-lama.pt' not in (os.listdir(LAMA_MODEL_PATH)):
    fs = Filesplit()
    fs.merge(input_dir=LAMA_MODEL_PATH)

if 'inference.pdiparams' not in os.listdir(DET_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=DET_MODEL_PATH)

if 'ProPainter.pth' not in os.listdir(VIDEO_INPAINT_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=VIDEO_INPAINT_MODEL_PATH)

# Specify ffmpeg executable path
sys_str = platform.system()
if sys_str == "Windows":
    ffmpeg_bin = os.path.join('win_x64', 'ffmpeg.exe')
elif sys_str == "Linux":
    ffmpeg_bin = os.path.join('linux_x64', 'ffmpeg')
else:
    ffmpeg_bin = os.path.join('macos', 'ffmpeg')
FFMPEG_PATH = os.path.join(BASE_DIR, '', 'ffmpeg', ffmpeg_bin)

if 'ffmpeg.exe' not in os.listdir(os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64')):
    fs = Filesplit()
    fs.merge(input_dir=os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64'))
# Add executable permissions to ffmpeg
os.chmod(FFMPEG_PATH, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ×××××××××××××××××××× [Do not change] end ××××××××××××××××××××


@unique
class InpaintMode(Enum):
    """
    Image inpainting algorithm enumeration
    """
    STTN = 'sttn'
    LAMA = 'lama'
    PROPAINTER = 'propainter'


# ×××××××××××××××××××× [Can be changed] start ××××××××××××××××××××
# Whether to use h264 encoding, if you need to share the generated video on Android phones, please enable this option
USE_H264 = True

# ×××××××××× General settings start ××××××××××
"""
MODE optional algorithm types
- InpaintMode.STTN algorithm: Better effect for real person videos, fast speed, can skip subtitle detection
- InpaintMode.LAMA algorithm: Good effect for animated videos, average speed, cannot skip subtitle detection
- InpaintMode.PROPAINTER algorithm: Requires a lot of video memory, slow speed, good effect for videos with violent motion
"""
# [Set inpaint algorithm]
MODE = InpaintMode.PROPAINTER
# [Set pixel deviation]
# Used to determine if it is a non-subtitle area (generally it is believed that the length of subtitle text boxes should be greater than the width, if the height of the subtitle box is greater than the width, and the amplitude of the excess exceeds the specified pixel size, it is considered a wrong detection)
THRESHOLD_HEIGHT_WIDTH_DIFFERENCE = 10
# Used to enlarge the mask size to prevent the automatically detected text box from being too small, causing text edges or residues during the inpaint stage
SUBTITLE_AREA_DEVIATION_PIXEL = 20
# Used to determine whether two text boxes are in the same line of subtitles, if the height difference is within the specified pixel points, it is considered the same line
THRESHOLD_HEIGHT_DIFFERENCE = 20
# Used to determine whether the rectangular boxes of two subtitle texts are similar. If the X-axis and Y-axis deviations are within the specified threshold, it is considered the same text box
PIXEL_TOLERANCE_Y = 20  # Number of pixel points allowed for longitudinal deviation of the detection box
PIXEL_TOLERANCE_X = 20  # Number of pixel points allowed for horizontal deviation of the detection box
# ×××××××××× General settings end ××××××××××

# ×××××××××× InpaintMode.STTN algorithm settings start ××××××××××
# The following parameters only take effect when using the STTN algorithm
"""
1. STTN_SKIP_DETECTION
Meaning: Whether to use skip detection
Effect: Setting to True skips subtitle detection, which will save a lot of time, but may accidentally damage video frames without subtitles or cause the removed subtitles to be missed

2. STTN_NEIGHBOR_STRIDE
Meaning: Adjacent frame stride. If you need to fill in the missing area of the 50th frame, STTN_NEIGHBOR_STRIDE=5, then the algorithm will use the 45th frame, 40th frame, etc. as references.
Effect: Used to control the density of reference frame selection. A larger stride means using fewer and more dispersed reference frames, while a smaller stride means using more and more concentrated reference frames.

3. STTN_REFERENCE_LENGTH
Meaning: Parameter frame count. The STTN algorithm will look at the preceding and following frames of each frame to be repaired to obtain contextual information for repair.
Effect: Increasing this will increase video memory usage and improve processing effect, but slow down processing speed.

4. STTN_MAX_LOAD_NUM
Meaning: The maximum number of video frames that the STTN algorithm can load at a time
Effect: The larger the setting, the slower the speed, but the better the effect
Note: STTN_MAX_LOAD_NUM must be greater than STTN_NEIGHBOR_STRIDE and STTN_REFERENCE_LENGTH
"""
STTN_SKIP_DETECTION = True
# Reference frame stride
STTN_NEIGHBOR_STRIDE = 5
# Reference frame length (count)
STTN_REFERENCE_LENGTH = 10
# Set the maximum number of frames that the STTN algorithm can process simultaneously
STTN_MAX_LOAD_NUM = 50
if STTN_MAX_LOAD_NUM < STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE:
    STTN_MAX_LOAD_NUM = STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE
# ×××××××××× InpaintMode.STTN algorithm settings end ××××××××××

# ×××××××××× InpaintMode.PROPAINTER algorithm settings start ××××××××××
# [Set according to your GPU video memory size] The maximum number of images that can be processed simultaneously. The larger the setting, the better the processing effect, but the higher the video memory requirement.
# 1280x720p video setting 80 requires 25G video memory, setting 50 requires 19G video memory
# 720x480p video setting 80 requires 8G video memory, setting 50 requires 7G video memory
PROPAINTER_MAX_LOAD_NUM = 70
# ×××××××××× InpaintMode.PROPAINTER algorithm settings end ××××××××××

# ×××××××××× InpaintMode.LAMA algorithm settings start ××××××××××
# Whether to enable ultra-fast mode. After enabling, the inpaint effect is not guaranteed, and only the text in the text area will be removed
LAMA_SUPER_FAST = False
# ×××××××××× InpaintMode.LAMA algorithm settings end ××××××××××
# ×××××××××××××××××××× [Can be changed] end ××××××××××××××××××××