"""
SAM3 Video Model for FiftyOne - Text-based concept tracking for videos.

Finds and tracks ALL matching instances across video frames using text prompts.
"""

import logging
from typing import List, Dict, Optional

import numpy as np
import torch
from PIL import Image

import fiftyone as fo
from fiftyone import Model, SamplesMixin
from fiftyone.core.labels import Detection, Detections

logger = logging.getLogger(__name__)


def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Sam3VideoModel(Model, SamplesMixin):
    """
    Text-based concept tracking for videos using SAM3.
    
    Finds and tracks ALL instances matching text prompts across video frames.
    
    Usage:
        model = Sam3VideoModel(prompt="person")
        dataset.apply_model(model, label_field="tracked_people")
        
        # Or with per-sample prompts
        dataset.apply_model(
            model,
            label_field="results",
            prompt_field="text_prompt"  # Field with str or list
        )
    """
    
    def __init__(
        self,
        model_path: str = "facebook/sam3",
        prompt: str = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        max_frame_num_to_track: int = None,
        device: str = None,
        **kwargs
    ):
        """
        Initialize SAM3 Video model.
        
        Args:
            model_path: HuggingFace model ID or local path
            prompt: Default text prompt (single string or list)
            threshold: Confidence threshold for detections
            mask_threshold: Mask binarization threshold
            max_frame_num_to_track: Max frames to process (None = all)
            device: Device to use (cuda/cpu/mps, None for auto)
        """
        SamplesMixin.__init__(self)
        
        self._fields = {}
        self.model_path = model_path
        self.prompt = prompt
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.max_frame_num_to_track = max_frame_num_to_track
        
        # Setup device
        if device is None:
            device = get_device()
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load SAM3 Video model and processor."""
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        
        logger.info("Loading SAM3 Video model")
        
        self.model = Sam3VideoModel.from_pretrained("facebook/sam3").to(
            self.device,
            dtype=torch.bfloat16
        )
        self.processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
        
        self.model.eval()
        logger.info("SAM3 Video model loaded successfully")
    
    @property
    def media_type(self):
        return "video"
    
    @property
    def needs_fields(self):
        return self._fields
    
    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields
    
    def predict(self, video_reader, sample):
        """
        Process video and return frame-level detections.
        
        Args:
            video_reader: FiftyOne video reader
            sample: FiftyOne sample with video metadata
            
        Returns:
            Dict mapping frame_number → fo.Detections
        """
        # Extract text prompt
        text_prompt = self._get_text_prompt(sample)
        
        # Load video frames
        video_frames, frame_width, frame_height = self._load_video_frames(video_reader)
        
        logger.info(f"Loaded {len(video_frames)} frames ({frame_width}x{frame_height})")
        
        # Check for multi-concept
        if isinstance(text_prompt, list):
            return self._process_multi_concept(
                video_frames, text_prompt, frame_width, frame_height
            )
        else:
            return self._process_single_concept(
                video_frames, text_prompt, frame_width, frame_height
            )
    
    def _get_text_prompt(self, sample):
        """Extract text prompt from sample or use default."""
        if "prompt_field" in self._fields:
            field_name = self._fields["prompt_field"]
            if sample.has_field(field_name):
                prompt = sample[field_name]
                if prompt is not None:
                    return prompt
        
        return self.prompt if self.prompt else "object"
    
    def _load_video_frames(self, video_reader):
        """
        Load all video frames into memory.
        
        Args:
            video_reader: FiftyOne video reader
            
        Returns:
            Tuple of (frames, width, height)
        """
        frames = []
        frame_width, frame_height = video_reader.frame_size
        
        for frame in video_reader:
            # Convert numpy BGR to PIL RGB
            frame_rgb = frame[:, :, ::-1]  # BGR to RGB
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)
        
        return frames, frame_width, frame_height
    
    def _process_single_concept(
        self,
        video_frames,
        text_prompt,
        frame_width,
        frame_height
    ):
        """
        Process video with single text concept.
        
        Args:
            video_frames: List of PIL Images
            text_prompt: Single text string
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            Dict {frame_number: fo.Detections}
        """
        # Initialize inference session
        inference_session = self.processor.init_video_session(
            video=video_frames,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=torch.bfloat16
        )
        
        # Add text prompt
        inference_session = self.processor.add_text_prompt(
            inference_session=inference_session,
            text=text_prompt
        )
        
        logger.info(f"Tracking '{text_prompt}' across {len(video_frames)} frames")
        
        # Propagate through video
        frame_results = {}
        
        for model_outputs in self.model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=self.max_frame_num_to_track
        ):
            # Post-process outputs
            processed = self.processor.postprocess_outputs(
                inference_session, model_outputs
            )
            
            # Convert to FiftyOne format
            frame_idx = model_outputs.frame_idx
            frame_number = frame_idx + 1  # FiftyOne uses 1-indexed
            
            detections = self._convert_frame_output_to_detections(
                processed,
                frame_width,
                frame_height,
                label=text_prompt
            )
            
            frame_results[frame_number] = detections
        
        logger.info(f"Tracked objects in {len(frame_results)} frames")
        
        return frame_results
    
    def _process_multi_concept(
        self,
        video_frames,
        text_prompts,
        frame_width,
        frame_height
    ):
        """
        Process video with multiple text concepts.
        Runs one pass per concept and merges results.
        
        Args:
            video_frames: List of PIL Images
            text_prompts: List of text strings
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            Dict {frame_number: fo.Detections}
        """
        logger.info(f"Multi-concept tracking: {text_prompts}")
        
        # Process each concept separately
        all_concept_results = {}
        
        for concept in text_prompts:
            logger.info(f"Processing concept: '{concept}'")
            
            concept_results = self._process_single_concept(
                video_frames,
                concept,
                frame_width,
                frame_height
            )
            
            # Store by concept
            all_concept_results[concept] = concept_results
        
        # Merge results by frame
        merged_results = {}
        
        for frame_num in range(1, len(video_frames) + 1):
            all_detections = []
            
            # Collect detections from each concept for this frame
            for concept, concept_results in all_concept_results.items():
                if frame_num in concept_results:
                    all_detections.extend(concept_results[frame_num].detections)
            
            merged_results[frame_num] = Detections(detections=all_detections)
        
        logger.info(
            f"Multi-concept complete. Total: "
            f"{sum(len(d.detections) for d in merged_results.values())} detections"
        )
        
        return merged_results
    
    def _convert_frame_output_to_detections(
        self,
        frame_output: dict,
        frame_width: int,
        frame_height: int,
        label: str
    ) -> Detections:
        """
        Convert SAM3 video frame output to FiftyOne Detections.
        
        Args:
            frame_output: Dict with object_ids, masks, boxes, scores
            frame_width: Frame width
            frame_height: Frame height
            label: Concept label
            
        Returns:
            fo.Detections with tracking indices
        """
        detections = []
        
        object_ids = frame_output['object_ids'].cpu().numpy()
        masks = frame_output['masks'].cpu().numpy()
        boxes = frame_output['boxes'].cpu().numpy()
        scores = frame_output['scores'].cpu().numpy()
        
        for i, obj_id in enumerate(object_ids):
            # Convert boxes from xyxy absolute to relative xywh
            x1, y1, x2, y2 = boxes[i]
            
            rel_bbox = [
                x1 / frame_width,
                y1 / frame_height,
                (x2 - x1) / frame_width,
                (y2 - y1) / frame_height
            ]
            
            # Crop mask to bounding box
            mask = masks[i]
            y1_int, y2_int = int(round(y1)), int(round(y2))
            x1_int, x2_int = int(round(x1)), int(round(x2))
            cropped_mask = mask[y1_int:y2_int, x1_int:x2_int]
            
            detection = Detection(
                label=label,
                bounding_box=rel_bbox,
                mask=cropped_mask,
                confidence=float(scores[i]),
                index=int(obj_id)  # SAM3's object ID for tracking
            )
            detections.append(detection)
        
        return Detections(detections=detections)

