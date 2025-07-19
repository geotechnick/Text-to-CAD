#!/usr/bin/env python3
"""
Simple test script for the Text-to-CAD Multi-Agent System
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_basic_functionality():
    """Test basic system functionality"""
    
    print("🧪 Testing Text-to-CAD Multi-Agent System...")
    
    try:
        from src.main import get_system
        
        # Initialize system
        print("🚀 Initializing system...")
        system = get_system()
        await system.initialize()
        print("✅ System initialized successfully")
        
        # Test prompt processing
        print("🔍 Testing prompt processing...")
        result = await system.process_prompt(
            "Design a simple concrete wall 3m high and 10m long",
            []
        )
        print(f"✅ Prompt processed: {result['status']}")
        
        # Test system status
        print("📊 Getting system status...")
        status = await system.get_system_status()
        print(f"✅ System status: {len(status.get('agents', {}))} agents running")
        
        # Shutdown
        print("🛑 Shutting down system...")
        await system.shutdown()
        print("✅ System shutdown complete")
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Fix Windows multiprocessing
    import multiprocessing as mp
    mp.freeze_support()
    
    success = asyncio.run(test_basic_functionality())
    sys.exit(0 if success else 1)