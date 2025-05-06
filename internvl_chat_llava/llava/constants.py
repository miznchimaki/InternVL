CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# Constants for Qwen2 Tokenizer (IternVL-2.5 & InternVL-3)
IMG_START_TOKEN = "<img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_END_TOKEN = "</img>"

# Constants for image processor & dynamic resolution strategy
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
