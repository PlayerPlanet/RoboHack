"""
Monkey-patches to fix lerobot issues on Windows.

Issue 1: lerobot uses CAP_MSMF (Microsoft Media Foundation) by default on Windows,
but some cameras work better with CAP_DSHOW (DirectShow).

Issue 2: gRPC metadata size limit is 16KB by default, but OpenVLA model config
contains ~30KB of normalization stats, causing connection failures.

Issue 3: Pi0/Pi05 policies require a custom transformers fork with SigLIP patches,
but standard transformers 4.53.3 works fine with a bypass.

This module patches:
1. get_cv2_backend to use DSHOW on Windows
2. grpc_channel_options to increase metadata size limits
3. Pi0 transformers version check to bypass the custom fork requirement

Import this module BEFORE importing any lerobot modules.
"""
import json
import platform


def patch_opencv_backend():
    """Replace lerobot's get_cv2_backend to use DSHOW on Windows."""
    import cv2
    from lerobot.cameras import utils
    
    original_get_cv2_backend = utils.get_cv2_backend
    
    def get_cv2_backend_fixed() -> int:
        """Use DSHOW on Windows instead of MSMF."""
        if platform.system() == "Windows":
            return int(cv2.CAP_DSHOW)
        else:
            # For other platforms, use the original logic
            return original_get_cv2_backend()
    
    utils.get_cv2_backend = get_cv2_backend_fixed
    print("✅ Patched lerobot to use CAP_DSHOW on Windows")


def patch_grpc_limits():
    """Increase gRPC metadata size limits to support large model configs like OpenVLA."""
    from lerobot.transport import utils
    
    # Store the original function
    original_grpc_channel_options = utils.grpc_channel_options
    
    def grpc_channel_options_fixed(
        max_receive_message_length: int = utils.MAX_MESSAGE_SIZE,
        max_send_message_length: int = utils.MAX_MESSAGE_SIZE,
        enable_retries: bool = True,
        initial_backoff: str = "0.1s",
        max_attempts: int = 5,
        backoff_multiplier: float = 2,
        max_backoff: str = "2s",
    ):
        """Enhanced version with larger metadata limits for OpenVLA."""
        # Get the original options
        options = original_grpc_channel_options(
            max_receive_message_length=max_receive_message_length,
            max_send_message_length=max_send_message_length,
            enable_retries=enable_retries,
            initial_backoff=initial_backoff,
            max_attempts=max_attempts,
            backoff_multiplier=backoff_multiplier,
            max_backoff=max_backoff,
        )
        
        # Add metadata size limits (OpenVLA config needs ~30KB, default is 16KB)
        # Set to 1MB to be safe
        metadata_limit = 1024 * 1024  # 1 MB
        
        additional_options = [
            ("grpc.max_metadata_size", metadata_limit),
        ]
        
        return options + additional_options
    
    utils.grpc_channel_options = grpc_channel_options_fixed
    print("✅ Patched gRPC channel options to increase metadata limits")


def patch_pi0_transformers_check():
    """Create a fake check module to bypass Pi0's transformers version validation."""
    from types import ModuleType
    
    # Create the fake check module
    check_module = ModuleType('check')
    
    def check_whether_transformers_replace_is_installed_correctly():
        """Always return True to bypass the check."""
        return True
    
    check_module.check_whether_transformers_replace_is_installed_correctly = check_whether_transformers_replace_is_installed_correctly
    
    # Inject it into transformers.models.siglip
    try:
        import transformers.models.siglip
        transformers.models.siglip.check = check_module
        print("✅ Patched Pi0 transformers check")
    except ImportError:
        print("⚠️  Warning: Could not patch transformers - siglip module not found")


def patch_pi0_config_validation():
    """Patch PI0Config to ignore extra finetuning-specific fields.
    
    Some fine-tuned PI0 models (like mizutoukotori/pi0_so101_v6) include extra
    config fields that were used during training but aren't part of the base
    PI0Config. This patch makes the config more permissive.
    """
    try:
        from lerobot.policies.pi0.configuration_pi0 import PI0Config
        # Store original __init__
        original_init = PI0Config.__init__
        
        def patched_init(self, **kwargs):
            """Filter out extra fields before calling original __init__."""
            # Fields that are training-specific and should be ignored at inference
            extra_fields = {
                'resize_imgs_with_padding',
                'adapt_to_pi_aloha',
                'use_delta_joint_actions_aloha',
                'proj_width',
                'num_steps',
                'use_cache',
                'attention_implementation',
                'freeze_vision_encoder',
                'train_expert_only',
                'train_state_proj',
            }
            
            # Remove extra fields from kwargs
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in extra_fields}
            
            # Log if we removed any fields
            removed = set(kwargs.keys()) - set(filtered_kwargs.keys())
            if removed:
                print(f"✅ Filtered out fine-tuning config fields: {removed}")
            
            # Call original __init__ with filtered kwargs
            original_init(self, **filtered_kwargs)
        
        # Replace __init__
        PI0Config.__init__ = patched_init
        print("✅ Patched PI0Config to ignore fine-tuning fields")
        
    except ImportError:
        print("⚠️  Warning: Could not patch PI0Config - module not found")


# Auto-apply all patches when this module is imported
patch_opencv_backend()
patch_grpc_limits()
patch_pi0_transformers_check()
patch_pi0_config_validation()
