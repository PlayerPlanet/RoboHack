"""
Patch to add GR00T support to LeRobot's async policy server.

CRITICAL ISSUE DISCOVERED:
- lerobot 0.4.0's GrootConfig is INCOMPATIBLE with nvidia/GR00T-N1.5-3B
- The config fields don't match (lerobot expects different fields)
- We need to bypass lerobot's GrootConfig and load the nvidia model directly

This patch:
1. Registers 'groot' in SUPPORTED_POLICIES
2. Patches get_policy_class to return a wrapper that loads nvidia GR00T directly
"""

def patch_groot_policy_support():
    """Add groot support by wrapping nvidia's GR00T model."""
    try:
        from lerobot.async_inference import policy_server
        
        # First, add groot to supported policies
        if 'groot' not in policy_server.SUPPORTED_POLICIES:
            policy_server.SUPPORTED_POLICIES.append('groot')
            print(f"✅ Added 'groot' to SUPPORTED_POLICIES")
        
        # Patch get_policy_class to handle groot specially
        original_get_policy_class = policy_server.get_policy_class
        
        def patched_get_policy_class(policy_type: str):
            """Return policy class, with special handling for groot."""
            if policy_type == 'groot':
                # Import nvidia's GR00T directly, bypassing lerobot's GrootConfig
                try:
                    from transformers import AutoModelForCausalLM, AutoConfig
                    import torch
                    
                    class NvidiaGrootWrapper:
                        """Wrapper to make nvidia/GR00T-N1.5-3B compatible with lerobot's async server."""
                        
                        def __init__(self, pretrained_name_or_path: str, **kwargs):
                            """Load nvidia GR00T model using transformers directly."""
                            print(f"🤖 Loading nvidia GR00T from {pretrained_name_or_path}")
                            
                            # Load config and model using transformers
                            config = AutoConfig.from_pretrained(
                                pretrained_name_or_path,
                                trust_remote_code=True
                            )
                            
                            device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
                            
                            self.model = AutoModelForCausalLM.from_pretrained(
                                pretrained_name_or_path,
                                config=config,
                                trust_remote_code=True,
                                torch_dtype=torch.bfloat16,
                                device_map=device
                            )
                            
                            self.device = device
                            self.config = config
                            
                            print(f"✅ Loaded GR00T model on {device}")
                        
                        def __call__(self, observation, **kwargs):
                            """Run inference - adapt observation format to GR00T's expected input."""
                            # TODO: Implement proper observation -> GR00T input conversion
                            # For now, return a placeholder
                            import torch
                            # GR00T expects action_dim=32, action_horizon=16 by default
                            action_dim = self.config.action_dim if hasattr(self.config, 'action_dim') else 32
                            action_horizon = self.config.action_horizon if hasattr(self.config, 'action_horizon') else 16
                            
                            # Return dummy actions for now (TODO: implement actual inference)
                            return torch.zeros((action_horizon, action_dim), device=self.device)
                        
                        def reset(self):
                            """Reset policy state if needed."""
                            pass
                    
                    return NvidiaGrootWrapper
                    
                except Exception as e:
                    print(f"❌ Failed to create GR00T wrapper: {e}")
                    raise
            else:
                # Use original function for other policy types
                return original_get_policy_class(policy_type)
        
        policy_server.get_policy_class = patched_get_policy_class
        print(f"✅ Patched get_policy_class to support nvidia GR00T")
        print(f"   Supported policies: {policy_server.SUPPORTED_POLICIES}")
        
    except Exception as e:
        print(f"⚠️  Could not patch groot support: {e}")
        import traceback
        traceback.print_exc()


# Auto-apply when imported
if __name__ != "__main__":
    patch_groot_policy_support()
