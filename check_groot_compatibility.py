#!/usr/bin/env python3
"""
Check what GR00T models lerobot actually supports.
Run this on the server to see compatible repos.
"""

import sys

print("=" * 70)
print("GR00T Compatibility Check")
print("=" * 70)
print()

try:
    from lerobot.policies.groot import GrootConfig, GrootPolicy
    print("✅ GrootPolicy and GrootConfig imported")
    
    # Check what fields GrootConfig actually expects
    print("\nGrootConfig expected fields:")
    import inspect
    sig = inspect.signature(GrootConfig.__init__)
    params = list(sig.parameters.keys())
    print(f"   Parameters: {params}")
    
    # Try to see if there are any example configs or default values
    try:
        default_config = GrootConfig()
        print("\n✅ Created default GrootConfig")
        print(f"   Config attributes: {sorted([a for a in dir(default_config) if not a.startswith('_')])[:20]}")
    except Exception as e:
        print(f"\n❌ Cannot create default GrootConfig: {e}")
    
    # Check if there's a model_type or repo_id hint
    if hasattr(GrootConfig, '__doc__'):
        print(f"\nGrootConfig docstring:")
        print(f"   {GrootConfig.__doc__}")
    
    # Try to find any references to pretrained models
    import lerobot.policies.groot as groot_module
    module_doc = groot_module.__doc__
    if module_doc:
        print(f"\nGroot module docstring:")
        print(f"   {module_doc}")
    
    # Check the source file for any repo_id references
    print(f"\nGroot module file: {groot_module.__file__}")
    
    # Try loading from lerobot's own model repo if it exists
    print("\n" + "=" * 70)
    print("Testing common GR00T model repositories:")
    print("=" * 70)
    
    test_repos = [
        "lerobot/groot",
        "lerobot/groot_v1",
        "nvidia/GR00T-N1.5-3B",
        "huggingface/groot",
    ]
    
    for repo in test_repos:
        try:
            print(f"\nTrying: {repo}")
            config = GrootConfig.from_pretrained(repo)
            print(f"   ✅ SUCCESS! {repo} is compatible")
            print(f"   Config type: {type(config)}")
            break
        except FileNotFoundError:
            print(f"   ❌ Repository not found")
        except Exception as e:
            error_msg = str(e)
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + "..."
            print(f"   ❌ Failed: {error_msg}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Recommendation:")
print("=" * 70)
print("""
If nvidia/GR00T-N1.5-3B is incompatible with lerobot 0.4.0's GrootConfig,
you have three options:

1. Use a different policy that IS compatible (pi0, pi05, smolvla, etc.)
2. Upgrade lerobot to a version that supports GR00T N1.5
3. Downgrade to an older GR00T model that lerobot 0.4.0 supports
4. Use the model directly without lerobot's GrootConfig wrapper

Checking lerobot's supported models...
""")

try:
    from lerobot.policies import get_policy_class
    print("\nAvailable policies in lerobot 0.4.0:")
    policies = ['act', 'smolvla', 'diffusion', 'tdmpc', 'vqbet', 'pi0', 'pi05', 'groot']
    for policy_name in policies:
        try:
            policy_class = get_policy_class(policy_name)
            print(f"   ✅ {policy_name}: {policy_class.__name__}")
        except Exception as e:
            print(f"   ❌ {policy_name}: Not available")
except Exception as e:
    print(f"Could not list policies: {e}")

print("\n" + "=" * 70)
