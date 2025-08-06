#!/usr/bin/env python3
"""
Test compression implementation for persistence diagrams
"""

import numpy as np
import zstd
import json
import time
from redis_store import COMPRESSION_LEVEL, COMPRESSION_THRESHOLD

def test_compression():
    """Test that compression/decompression works correctly."""
    
    # Create a test persistence diagram
    num_points = 500  # Typical size
    diagram = np.random.rand(num_points, 2).astype(np.float32)
    
    print(f"Original diagram shape: {diagram.shape}")
    print(f"Original diagram dtype: {diagram.dtype}")
    
    # Convert to bytes
    diagram_bytes = diagram.tobytes()
    original_size = len(diagram_bytes)
    print(f"Original size: {original_size} bytes")
    
    # Compress
    start = time.perf_counter()
    compressed = zstd.compress(diagram_bytes, COMPRESSION_LEVEL)
    compress_time = (time.perf_counter() - start) * 1000
    compressed_size = len(compressed)
    
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {compressed_size/original_size:.1%}")
    print(f"Compression time: {compress_time:.2f}ms")
    
    # Convert to hex for storage (like in Redis)
    compressed_hex = compressed.hex()
    print(f"Hex string size: {len(compressed_hex)} chars")
    
    # Decompress
    start = time.perf_counter()
    compressed_bytes = bytes.fromhex(compressed_hex)
    decompressed = zstd.decompress(compressed_bytes)
    decompress_time = (time.perf_counter() - start) * 1000
    
    print(f"Decompression time: {decompress_time:.2f}ms")
    
    # Reconstruct array
    reconstructed = np.frombuffer(decompressed, dtype=np.float32).reshape(diagram.shape)
    
    # Verify
    if np.array_equal(diagram, reconstructed):
        print("✅ Compression/decompression successful - arrays match perfectly!")
    else:
        print("❌ ERROR: Arrays don't match!")
        print(f"Max difference: {np.max(np.abs(diagram - reconstructed))}")
    
    # Test with different sizes
    print("\n" + "="*50)
    print("Testing different diagram sizes:")
    print("="*50)
    
    for num_points in [10, 50, 100, 500, 1000, 5000]:
        diagram = np.random.rand(num_points, 2).astype(np.float32)
        original_size = len(diagram.tobytes())
        
        if original_size > COMPRESSION_THRESHOLD:
            compressed = zstd.compress(diagram.tobytes(), COMPRESSION_LEVEL)
            ratio = len(compressed) / original_size
            print(f"Points: {num_points:5d} | Size: {original_size:7d} | Compressed: {len(compressed):7d} | Ratio: {ratio:.1%}")
        else:
            print(f"Points: {num_points:5d} | Size: {original_size:7d} | Below threshold - not compressed")
    
    # Test metadata structure
    print("\n" + "="*50)
    print("Testing metadata structure:")
    print("="*50)
    
    metadata = {
        "persistence_diagram": diagram.tolist(),
        "other_field": "test"
    }
    
    # Simulate compression logic
    if "persistence_diagram" in metadata:
        diagram_array = np.array(metadata["persistence_diagram"], dtype=np.float32)
        diagram_bytes = diagram_array.tobytes()
        
        if len(diagram_bytes) > COMPRESSION_THRESHOLD:
            compressed = zstd.compress(diagram_bytes, COMPRESSION_LEVEL)
            metadata["persistence_diagram_compressed"] = compressed.hex()
            metadata["persistence_diagram_shape"] = diagram_array.shape
            metadata["persistence_diagram_dtype"] = str(diagram_array.dtype)
            del metadata["persistence_diagram"]
    
    # Check JSON serialization
    try:
        json_str = json.dumps(metadata)
        print(f"✅ Metadata serializes to JSON successfully")
        print(f"JSON size: {len(json_str)} bytes")
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")


if __name__ == "__main__":
    test_compression()