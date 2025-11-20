#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

try:
    # Test imports
    from models.simplecil_knn import Learner
    from utils.factory import get_model
    print("✓ All imports successful")

    # Test model creation
    args = {
        "model_name": "simplecil_knn",
        "backbone_type": "pretrained_vit_b16_224",
        "device": ["cpu"],  # Use CPU for testing
        "memory_size": 0,
        "memory_per_class": 0,
        "fixed_memory": False,
        "init_cls": 10,
        "increment": 10,
        "seed": 1993,
    }

    print("✓ Creating SimpleCIL-KNN model...")
    model = get_model("simplecil_knn", args)
    print("✓ Model created successfully")
    print(f"✓ Model type: {type(model._network)}")

    print("✓ SimpleCIL-KNN basic functionality test passed!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
