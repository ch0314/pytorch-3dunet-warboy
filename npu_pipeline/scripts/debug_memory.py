#!/usr/bin/env python3
import psutil
import os
import time
import gc
import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def monitor_memory(config_path, duration=60):
    """Monitor memory usage during pipeline execution"""
    process = psutil.Process(os.getpid())
    
    print(f"🔍 Memory monitoring for {duration} seconds...")
    print(f"💾 Initial memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    print(f"💾 Available memory: {psutil.virtual_memory().available / 1024 / 1024:.1f} MB")
    
    # Load config to check data size
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Quick data size estimation
    from src.loader import H5Loader
    loader = H5Loader()
    data_dict = loader.load_from_config(config)
    
    if 'test' in data_dict:
        for i, data_info in enumerate(data_dict['test']):
            volume = data_info['data']
            volume_size = volume.nbytes / 1024 / 1024  # MB
            print(f"📦 Test volume {i+1}: {volume.shape} ({volume_size:.1f} MB)")
    
    # Memory monitoring loop
    max_memory = 0
    start_time = time.time()
    
    while time.time() - start_time < duration:
        current_memory = process.memory_info().rss / 1024 / 1024
        max_memory = max(max_memory, current_memory)
        
        print(f"💾 Current: {current_memory:.1f} MB, Peak: {max_memory:.1f} MB", end='\r')
        time.sleep(1)
    
    print(f"\n🎯 Peak memory usage: {max_memory:.1f} MB")
    print(f"🎯 Memory increase: {max_memory - (process.memory_info().rss / 1024 / 1024):.1f} MB")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python debug_memory.py <config_path>")
        sys.exit(1)
    
    monitor_memory(sys.argv[1])