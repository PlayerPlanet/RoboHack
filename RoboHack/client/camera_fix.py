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
    PI0Config. This patch makes the config more permissive by intercepting
    the from_dict() method which is called when loading from HuggingFace.
    """
    PI0Config = None
    
    # Try different import paths (lerobot structure has changed over time)
    import_paths = [
        "lerobot.policies.pi0.configuration_pi0",
        "lerobot.policy.pi0.configuration_pi0",
        "lerobot.common.policies.pi0.configuration_pi0",
    ]
    
    for import_path in import_paths:
        try:
            parts = import_path.split('.')
            module_path = '.'.join(parts[:-1])
            class_name = parts[-1]
            
            module = __import__(module_path, fromlist=[class_name])
            PI0Config = getattr(module, 'PI0Config')
            print(f"✅ Found PI0Config at {import_path}")
            break
        except (ImportError, AttributeError):
            continue
    
    if PI0Config is None:
        print("⚠️  Warning: Could not find PI0Config in any known location")
        return
    
    # Fields that are training-specific and should be ignored at inference
    EXTRA_FIELDS = {
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
    
    # Patch __init__ to filter kwargs
    original_init = PI0Config.__init__
    
    def patched_init(self, **kwargs):
        """Filter out extra fields before calling original __init__."""
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in EXTRA_FIELDS}
        removed = set(kwargs.keys()) - set(filtered_kwargs.keys())
        if removed:
            print(f"✅ [__init__] Filtered out: {removed}")
        original_init(self, **filtered_kwargs)
    
    # Patch from_dict (this is what HuggingFace uses to load configs)
    if hasattr(PI0Config, 'from_dict'):
        original_from_dict = PI0Config.from_dict
        
        @classmethod
        def patched_from_dict(cls, config_dict, **kwargs):
            """Filter out extra fields from config_dict before validation."""
            if isinstance(config_dict, dict):
                filtered_dict = {k: v for k, v in config_dict.items() if k not in EXTRA_FIELDS}
                removed = set(config_dict.keys()) - set(filtered_dict.keys())
                if removed:
                    print(f"✅ [from_dict] Filtered out: {removed}")
                return original_from_dict(filtered_dict, **kwargs)
            return original_from_dict(config_dict, **kwargs)
        
        PI0Config.from_dict = patched_from_dict
    
    # Also patch the parent class's attribute validation if it exists
    # This catches validation that happens in PretrainedConfig
    try:
        from transformers.configuration_utils import PretrainedConfig
        if hasattr(PretrainedConfig, '__init__'):
            original_pretrained_init = PretrainedConfig.__init__
            
            def patched_pretrained_init(self, **kwargs):
                """Filter extra fields before PretrainedConfig validation."""
                # Only filter if this is being called from PI0Config
                if self.__class__.__name__ == 'PI0Config':
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in EXTRA_FIELDS}
                    removed = set(kwargs.keys()) - set(filtered_kwargs.keys())
                    if removed:
                        print(f"✅ [PretrainedConfig] Filtered out: {removed}")
                    original_pretrained_init(self, **filtered_kwargs)
                else:
                    original_pretrained_init(self, **kwargs)
            
            PretrainedConfig.__init__ = patched_pretrained_init
    except ImportError:
        pass
    
    PI0Config.__init__ = patched_init
    print("✅ Patched PI0Config.__init__ and from_dict() to ignore fine-tuning fields")


# Auto-apply all patches when this module is imported
patch_opencv_backend()
patch_grpc_limits()
patch_pi0_transformers_check()
patch_pi0_config_validation()
