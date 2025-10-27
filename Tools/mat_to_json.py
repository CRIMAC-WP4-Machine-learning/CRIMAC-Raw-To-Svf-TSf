#!/usr/bin/env python3
import sys
import json
import numpy as np
from scipy.io import loadmat
from pathlib import Path

def make_serializable(obj):
    """Recursively convert NumPy/MATLAB data to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {"real": obj.real.tolist(), "imag": obj.imag.tolist()}
        else:
            return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="ignore")
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    else:
        return str(obj)

def convert_mat_to_json(mat_filename, json_filename):
    """Convert a MATLAB .mat file to a JSON file."""
    mat_data = loadmat(mat_filename)

    # Remove MATLAB metadata
    clean_data = {k: v for k, v in mat_data.items() if not k.startswith("__")}

    # Convert to serializable format
    serializable = {k: make_serializable(v) for k, v in clean_data.items()}

    # Write to JSON
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"✅ Converted '{mat_filename}' → '{json_filename}'")

def main():
    if len(sys.argv) < 2:
        print("Usage: python mat_to_json.py <input.mat> [output.json]")
        sys.exit(1)

    mat_file = Path(sys.argv[1])
    if not mat_file.exists():
        print(f"❌ File not found: {mat_file}")
        sys.exit(1)

    # Default output file name
    json_file = Path(sys.argv[2]) if len(sys.argv) > 2 else mat_file.with_suffix(".json")

    convert_mat_to_json(mat_file, json_file)

if __name__ == "__main__":
    main()
