import logging

from huggingface_hub import snapshot_download
from fiftyone.operators import types

from .zoo import Sam3VideoModel

logger = logging.getLogger(__name__)


def download_model(model_name, model_path):
    """Downloads the SAM3 model from HuggingFace.
    
    Args:
        model_name: the name of the model to download
        model_path: the absolute filename or directory to download to
    """
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(model_name=None, model_path=None, **kwargs):
    """Load SAM3 Video model for text-based concept tracking.
    
    Args:
        model_name: Model name (unused, for compatibility)
        model_path: HuggingFace model ID (default: "facebook/sam3")
        **kwargs: Additional parameters:
            - prompt: Default text prompt (str or list)
            - threshold: Confidence threshold (default: 0.5)
            - mask_threshold: Mask binarization threshold (default: 0.5)
            - max_frame_num_to_track: Max frames to process (default: None)
            - device: Device to use (cuda/cpu/mps, None for auto)
    
    Returns:
        Sam3VideoModel: Initialized model
    
    Example:
        model = load_model(prompt="person")
        dataset.apply_model(model, label_field="people")
        
        # Multi-concept
        model = load_model(prompt=["person", "car", "dog"])
        dataset.apply_model(model, label_field="tracked_objects")
    """

    
    return Sam3VideoModel(model_path="facebook/sam3", **kwargs)


def resolve_input(model_name, ctx):
    """Defines properties to collect the model's custom parameters."""
    inputs = types.Object()
    
    # Text prompt
    inputs.str(
        "prompt",
        default=None,
        required=False,
        label="Text Prompt",
        description="Text prompt for concept tracking (e.g., 'person', 'car')",
    )
    
    # Thresholds
    inputs.float(
        "threshold",
        default=0.5,
        label="Confidence Threshold",
        description="Minimum confidence score for detections",
    )
    
    inputs.float(
        "mask_threshold",
        default=0.5,
        label="Mask Threshold",
        description="Threshold for mask binarization",
    )
    
    # Frame limit
    inputs.int(
        "max_frame_num_to_track",
        default=None,
        required=False,
        label="Max Frames",
        description="Maximum number of frames to process (None = all)",
    )
    
    # Device
    inputs.enum(
        "device",
        values=["auto", "cuda", "cpu", "mps"],
        default="auto",
        label="Device",
        description="Device to use for inference",
    )
    
    return types.Property(inputs)
