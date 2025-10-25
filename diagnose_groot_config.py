#!/usr/bin/env python3
"""
Diagnostic script to understand GR00T config serialization issues.
Run this on the server to see how configs are structured.
"""

import sys

print("=" * 70)
print("GR00T Configuration Diagnostic")
print("=" * 70)
print()

# Check lerobot version and imports
try:
    import lerobot
    print(f"✅ lerobot version: {lerobot.__version__}")
except Exception as e:
    print(f"❌ Failed to import lerobot: {e}")
    sys.exit(1)

# Check GrootConfig
try:
    from lerobot.policies.groot import GrootConfig
    print(f"✅ GrootConfig imported")
    
    # Check if it's a PreTrainedConfig
    from lerobot.configs.policies import PreTrainedConfig
    print(f"   Is GrootConfig a PreTrainedConfig? {issubclass(GrootConfig, PreTrainedConfig)}")
    
    # Check model_type attribute
    if hasattr(GrootConfig, 'model_type'):
        print(f"   GrootConfig.model_type = {GrootConfig.model_type}")
    else:
        print(f"   ⚠️  GrootConfig has no model_type attribute")
    
    # Try to load from pretrained
    print(f"\n   Attempting to load GR00T config from HuggingFace...")
    try:
        config = GrootConfig.from_pretrained("nvidia/GR00T-N1.5-3B")
        print(f"   ✅ Loaded config from nvidia/GR00T-N1.5-3B")
        print(f"   Config type: {type(config)}")
        print(f"   Config dict keys: {list(config.to_dict().keys())[:10]}...")
        
        # Check for 'type' key
        config_dict = config.to_dict()
        if 'type' in config_dict:
            print(f"   ✅ Config dict has 'type' key: {config_dict['type']}")
        else:
            print(f"   ❌ Config dict MISSING 'type' key!")
            print(f"   Available keys: {sorted(config_dict.keys())}")
            
        # Check model_type in dict
        if 'model_type' in config_dict:
            print(f"   Has 'model_type': {config_dict['model_type']}")
            
    except Exception as e:
        print(f"   ❌ Failed to load config: {e}")
        
except Exception as e:
    print(f"❌ Failed to import GrootConfig: {e}")

# Check async inference server structure
print(f"\n" + "=" * 70)
print("Async Inference Server Structure")
print("=" * 70)

try:
    from lerobot.async_inference import policy_server
    print(f"✅ policy_server module imported")
    
    # List all attributes
    attrs = [a for a in dir(policy_server) if not a.startswith('_')]
    print(f"   Module attributes: {attrs}")
    
    # Check for load_policy function
    if hasattr(policy_server, 'load_policy'):
        print(f"   ✅ Has load_policy function")
        import inspect
        sig = inspect.signature(policy_server.load_policy)
        print(f"      Signature: {sig}")
    else:
        print(f"   ❌ No load_policy function")
        
    # Check for policy loading in serve
    if hasattr(policy_server, 'serve'):
        print(f"   ✅ Has serve function")
        
    # Check SUPPORTED_POLICIES
    if hasattr(policy_server, 'SUPPORTED_POLICIES'):
        print(f"   SUPPORTED_POLICIES: {policy_server.SUPPORTED_POLICIES}")
    
except Exception as e:
    print(f"❌ Failed to inspect policy_server: {e}")

print("\n" + "=" * 70)
print("Diagnostic complete")
print("=" * 70)
