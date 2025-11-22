#!/usr/bin/env python3
"""
Quick test script for C++ ONNX backend
Tests basic speech-to-text functionality
"""
import sys
import time
from pathlib import Path

import requests

def test_transcription(base_url, audio_file):
    """Test audio transcription"""
    print(f"\n📝 Testing transcription with: {audio_file}")
    
    try:
        # Detect MIME type based on extension
        mime_type = 'audio/opus' if str(audio_file).endswith('.opus') else 'audio/wav'
        
        with open(audio_file, 'rb') as f:
            files = {'file': (audio_file, f, mime_type)}
            
            start_time = time.time()
            response = requests.post(
                f"{base_url}/process_audio",
                files=files,
                timeout=30
            )
            elapsed = time.time() - start_time
            
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Transcription successful in {elapsed:.2f}s")
            print(f"   Text: {result.get('text', 'N/A')}")
            print(f"   Processing time: {result.get('processing_time', 'N/A')}s")
            return True
        else:
            print(f"❌ Transcription failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return False

def main():
    # Configuration
    base_url = "http://localhost:8005"
    
    # Find test audio files (OPUS format)
    test_files = []
    
    # Check for golos debug dataset (10 files)
    golos_debug = Path("data/golos_10_debug_extracted/golos_10_debug/files")
    if golos_debug.exists():
        opus_files = list(golos_debug.glob("*.opus"))
        if opus_files:
            test_files.extend(sorted(opus_files)[:3])  # Take first 3 files
    
    # Check for golos 1k dataset as fallback
    if not test_files:
        golos_1k = Path("data/golos_1k_extracted/golos_1k/files")
        if golos_1k.exists():
            opus_files = list(golos_1k.glob("*.opus"))
            if opus_files:
                test_files.extend(sorted(opus_files)[:3])  # Take first 3 files
    
    if not test_files:
        print("❌ No test audio files found!")
        print("   Please extract golos dataset in data/ directory")
        return 1
    
    print(f"🎯 Testing C++ ONNX Backend at {base_url}")
    print(f"📁 Found {len(test_files)} test files")
    
    # Run transcription tests
    success_count = 0
    for audio_file in test_files:
        if test_transcription(base_url, str(audio_file)):
            success_count += 1
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"   Total tests: {len(test_files)}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {len(test_files) - success_count}")
    
    if success_count == len(test_files):
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())