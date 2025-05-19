import torch
import folder_paths
from ..xdit_manager import manager

class XfuserClipTextEncode:
    """
    Xfuser CLIP Text Encoder
    
    Encodes text prompts into model-compatible representations
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "default": "a small dog"}),
                "negative": ("STRING", {"multiline": True, "default": ""})
            }
        }
    
    RETURN_TYPES = ("XFUSER_POSITIVE", "XFUSER_NEGATIVE",)
    RETURN_NAMES = ("positive", "negative",)
    FUNCTION = "encode_text"
    CATEGORY = "conditioning"
    
    def encode_text(self, positive, negative=""):
        """
        Encode text prompts
        
        Args:
            positive: Positive prompt text
            negative: Negative prompt text
            
        Returns:
            Encoded prompts
        """
        # In this implementation, we simply pass the text directly
        # because the Xfuser pipeline handles text encoding internally
        
        print(f"Encoding text - Positive: {positive}")
        if negative:
            print(f"Encoding text - Negative: {negative}")
            
        return (positive, negative) 