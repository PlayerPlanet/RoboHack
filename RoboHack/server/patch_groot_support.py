"""
Patch to add GR00T support to LeRobot's async policy server.

The GrootPolicy exists in lerobot.policies.groot but isn't registered
in the async inference server's SUPPORTED_POLICIES list.

This patch registers it so the server recognizes policy_type="groot".
"""

def patch_groot_policy_support():
    """Add groot to the supported policies in the async inference server."""
    try:
        from lerobot.async_inference import policy_server
        from lerobot.policies.groot import GrootPolicy, GrootConfig
        
        # Check if groot is already supported
        if 'groot' in policy_server.SUPPORTED_POLICIES:
            print("✅ groot already in SUPPORTED_POLICIES")
        else:
            # Add groot to supported policies
            policy_server.SUPPORTED_POLICIES.append('groot')
            print(f"✅ Added groot to SUPPORTED_POLICIES")
        
        # Register the policy class mapping if needed
        if hasattr(policy_server, 'POLICY_CLASSES'):
            policy_server.POLICY_CLASSES['groot'] = GrootPolicy
        
        # Patch the config deserialization to handle GR00T configs
        # GR00T configs need a 'type' key for proper deserialization
        original_load_policy = policy_server.load_policy
        
        def patched_load_policy(config):
            """Wrap load_policy to fix GR00T config deserialization."""
            # If the config dict is missing 'type' and it's a groot policy, add it
            if hasattr(config, 'policy_config') and isinstance(config.policy_config, dict):
                if 'type' not in config.policy_config and config.policy_type == 'groot':
                    config.policy_config['type'] = 'groot'
            return original_load_policy(config)
        
        policy_server.load_policy = patched_load_policy
            
        print(f"✅ Patched async server to support groot")
        print(f"   Supported policies: {policy_server.SUPPORTED_POLICIES}")
        
    except ImportError as e:
        print(f"⚠️  Could not patch groot support: {e}")
        print(f"   GR00T policy may not work with async server")


# Auto-apply when imported
if __name__ != "__main__":
    patch_groot_policy_support()
