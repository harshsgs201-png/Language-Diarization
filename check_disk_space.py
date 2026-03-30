#!/usr/bin/env python
"""Quick disk space diagnosis and cleanup for Language Diarization project."""

import os
import shutil
from pathlib import Path

def format_bytes(bytes_val):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"

def get_dir_size(path):
    """Get total size of directory."""
    try:
        return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
    except:
        return 0

def check_disk_space():
    """Check disk space on all drives."""
    import shutil
    
    print("\n" + "="*60)
    print("DISK SPACE ANALYSIS")
    print("="*60)
    
    for drive in ['C:', 'D:', 'F:']:
        try:
            usage = shutil.disk_usage(drive)
            free_gb = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)
            used_gb = usage.used / (1024**3)
            
            print(f"\n{drive}:")
            print(f"  Total:  {total_gb:7.1f} GB")
            print(f"  Used:   {used_gb:7.1f} GB")
            print(f"  Free:   {free_gb:7.1f} GB")
        except:
            pass

def analyze_project_directories():
    """Analyze sizes of project directories."""
    print("\n" + "="*60)
    print("PROJECT DIRECTORY SIZES")
    print("="*60)
    
    dirs = {
        'data/raw': 'Raw audio files',
        'data/processed': 'Labels + embeddings',
        'src': 'Source code',
        'models': 'Model weights'
    }
    
    for dir_path, desc in dirs.items():
        if os.path.exists(dir_path):
            size = get_dir_size(dir_path)
            print(f"{dir_path:20s} ({desc:25s}): {format_bytes(size)}")
        else:
            print(f"{dir_path:20s} ({desc:25s}): [NOT FOUND]")

def identify_large_files():
    """Find largest files in workspace."""
    print("\n" + "="*60)
    print("LARGEST FILES IN WORKSPACE")
    print("="*60)
    
    large_files = []
    
    # Search in key directories
    for root_dir in ['.', os.path.expanduser('~/.cache')]:
        if not os.path.exists(root_dir):
            continue
        try:
            for root, dirs, files in os.walk(root_dir):
                # Skip some deep directories for speed
                dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache']]
                
                for file in files:
                    try:
                        fpath = os.path.join(root, file)
                        size = os.path.getsize(fpath)
                        if size > 50 * 1024 * 1024:  # > 50 MB
                            large_files.append((fpath, size))
                    except:
                        pass
        except:
            pass
    
    large_files.sort(key=lambda x: x[1], reverse=True)
    
    print("\nLarge files (>50 MB):")
    for fpath, size in large_files[:15]:
        print(f"  {format_bytes(size):12s}  {fpath}")

def cleanup_suggestions():
    """Suggest cleanup actions."""
    print("\n" + "="*60)
    print("CLEANUP SUGGESTIONS")
    print("="*60)
    
    suggestions = [
        ("Temp files", "C:\\Windows\\Temp\\*", 100),
        ("User temp", "C:\\Users\\harsh\\AppData\\Local\\Temp\\*", 500),
        ("HF cache (old)", "C:\\Users\\harsh\\.cache\\huggingface*", 1000),
        ("HF cache (new)", "C:\\Users\\harsh\\.cache\\huggingface_xlsr*", 1500),
        ("pip cache", "C:\\Users\\harsh\\AppData\\Local\\pip\\*", 100),
    ]
    
    print("\nPotential cleanup locations (estimated space):")
    for name, path, est_mb in suggestions:
        print(f"  {name:25s}  (~{est_mb:4d} MB)  {path}")

def main():
    print("\n" + "="*70)
    print("LANGUAGE DIARIZATION: DISK SPACE DIAGNOSTIC")
    print("="*70)
    
    check_disk_space()
    analyze_project_directories()
    identify_large_files()
    cleanup_suggestions()
    
    print("\n" + "="*70)
    print("SPACE NEEDED FOR FULL PIPELINE")
    print("="*70)
    print("  XLSR model download:     ~1.3 GB")
    print("  Extracted embeddings:    ~5.0 GB (5,167 files × 1 MB)")
    print("  Training buffer:         ~0.5 GB")
    print("  ------------------------------------------")
    print("  TOTAL REQUIRED:          ~7.0 GB")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
