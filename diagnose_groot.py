#!/usr/bin/env python3
"""
Diagnose GR00T installation on the server.
Run this inside the Docker container to see what's available.
"""

import sys
import importlib.metadata

print("=" * 70)
print("GR00T Installation Diagnostic")
print("=" * 70)
print()

# 1. Check lerobot version and extras
print("1. LeRobot Installation:")
try:
    lerobot_version = importlib.metadata.version("lerobot")
    print(f"   ✅ lerobot version: {lerobot_version}")
    
    # Check if groot extra is installed
    try:
        metadata = importlib.metadata.metadata("lerobot")
        print(f"   Metadata keys: {list(metadata.keys())[:5]}...")
    except Exception as e:
        print(f"   ⚠️  Could not read metadata: {e}")
        
except Exception as e:
    print(f"   ❌ lerobot not installed: {e}")

print()

# 2. Check FlashAttention
print("2. FlashAttention:")
try:
    import flash_attn
    print(f"   ✅ flash_attn version: {flash_attn.__version__}")
except ImportError as e:
    print(f"   ❌ flash_attn not available: {e}")

print()

# 3. Check groot module structure
print("3. GR00T Module Structure:")
try:
    import lerobot.policies.groot as groot_module
    print(f"   ✅ groot module path: {groot_module.__file__}")
    print(f"   Module contents: {dir(groot_module)}")
    
    # Check for specific classes
    if hasattr(groot_module, 'Gr00tPolicy'):
        print(f"   ✅ Gr00tPolicy class found")
    else:
        print(f"   ❌ Gr00tPolicy class NOT found")
        
    if hasattr(groot_module, 'GrootPolicy'):
        print(f"   ✅ GrootPolicy class found")
    else:
        print(f"   ❌ GrootPolicy class NOT found")
        
except ImportError as e:
    print(f"   ❌ Cannot import lerobot.policies.groot: {e}")

print()

# 4. Check supported policies in async server
print("4. Async Policy Server Support:")
try:
    from lerobot.async_inference.policy_server import SUPPORTED_POLICIES
    print(f"   Supported policies: {SUPPORTED_POLICIES}")
    
    if 'groot' in SUPPORTED_POLICIES:
        print(f"   ✅ 'groot' is in supported policies")
    else:
        print(f"   ❌ 'groot' is NOT in supported policies")
        
except ImportError as e:
    print(f"   ❌ Cannot import policy_server: {e}")

print()

# 5. Try to find all groot-related modules
print("5. All GR00T-related imports:")
groot_imports = [
    "lerobot.policies.groot",
    "lerobot.policies.groot.modeling_groot",
    "lerobot.policies.groot.configuration_groot",
]

for module_name in groot_imports:
    try:
        module = importlib.import_module(module_name)
        print(f"   ✅ {module_name}")
        if hasattr(module, '__all__'):
            print(f"      Exports: {module.__all__}")
    except ImportError as e:
        print(f"   ❌ {module_name}: {e}")

print()

# 6. Check installed packages for groot dependencies
print("6. Installed Packages (groot-related):")
for pkg in ['lerobot', 'flash-attn', 'torch', 'transformers']:
    try:
        version = importlib.metadata.version(pkg)
        print(f"   {pkg}: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"   {pkg}: NOT INSTALLED")

print()
print("=" * 70)
print("Recommendation:")
print("=" * 70)

# Final recommendation
try:
    from lerobot.async_inference.policy_server import SUPPORTED_POLICIES
    if 'groot' not in SUPPORTED_POLICIES:
        print("❌ GR00T is NOT supported by this lerobot version.")
        print("   Try: pip install --upgrade 'lerobot[groot]'")
    else:
        print("✅ GR00T should be working!")
except:
    print("⚠️  Cannot determine GR00T status")

print()
