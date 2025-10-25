"""
Server-side debugging patch to see what observations are actually received.
"""

def patch_server_observation_logging():
    """Add logging to see what observations the server actually receives."""
    try:
        from lerobot.async_inference import policy_server
        
        # Patch the policy server's observation handler
        if hasattr(policy_server, 'PolicyServer'):
            original_stream_actions = policy_server.PolicyServer.StreamActions
            
            async def patched_stream_actions(self, request_iterator, context):
                """Wrap StreamActions to log received observations."""
                print("=" * 70)
                print("🔍 SERVER: StreamActions called")
                print("=" * 70)
                
                # Wrap the request iterator to log observations
                async def logging_iterator():
                    async for request in request_iterator:
                        print(f"🔍 SERVER: Received request type: {type(request)}")
                        if hasattr(request, 'observation'):
                            print(f"🔍 SERVER: Observation keys: {list(request.observation.keys()) if hasattr(request.observation, 'keys') else 'N/A'}")
                        yield request
                
                # Call original with logging iterator
                return await original_stream_actions(self, logging_iterator(), context)
            
            policy_server.PolicyServer.StreamActions = patched_stream_actions
            print("✅ Patched PolicyServer to log received observations")
    
    except Exception as e:
        print(f"⚠️  Could not patch server logging: {e}")
        import traceback
        traceback.print_exc()


# Auto-apply
patch_server_observation_logging()
