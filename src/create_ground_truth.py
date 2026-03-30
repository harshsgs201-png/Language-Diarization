"""
Convert label .npy files to RTTM ground truth format for evaluation
"""
import numpy as np
import os

# Files to convert
files = ["AD09001", "AD09004", "AD09008"]
CLASS_MAP = {0: "English", 1: "Hindi", 2: "Other"}
HOP_LENGTH = 0.01  # 10ms per frame

output_dir = "data/raw/mini_eval/rttm"
label_dir = "data/processed"

for file_id in files:
    label_path = os.path.join(label_dir, f"{file_id}_labels.npy")
    rttm_path = os.path.join(output_dir, f"{file_id}.rttm")
    
    if not os.path.exists(label_path):
        print(f"⚠️  {label_path} not found, skipping...")
        continue
    
    # Load labels
    labels = np.load(label_path).astype(int)
    print(f"Converting {file_id}: {len(labels)} frames")
    
    # Convert to RTTM
    with open(rttm_path, 'w') as f:
        current_class = labels[0]
        start_frame = 0
        
        for i in range(1, len(labels)):
            if labels[i] != current_class:
                # Write segment
                start_time = start_frame * HOP_LENGTH
                end_time = i * HOP_LENGTH
                duration = end_time - start_time
                label_name = CLASS_MAP[current_class]
                
                f.write(f"SPEAKER {file_id} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {label_name} <NA> <NA>\n")
                
                current_class = labels[i]
                start_frame = i
        
        # Write final segment
        start_time = start_frame * HOP_LENGTH
        end_time = len(labels) * HOP_LENGTH
        duration = end_time - start_time
        label_name = CLASS_MAP[current_class]
        f.write(f"SPEAKER {file_id} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {label_name} <NA> <NA>\n")
    
    print(f"  ✓ Created {rttm_path}")

print("\n✅ Ground truth RTTM files created!")
