"""Environment validation script for the medical vision pipeline."""

import json
import sys
import platform
from datetime import datetime, timezone
from pathlib import Path


def validate_environment() -> dict:
    """Validate the environment and return environment info."""
    env_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    errors = []

    # Check Python version
    if sys.version_info < (3, 10):
        errors.append(f"Python 3.10+ required, got {platform.python_version()}")
    print(f"[OK] Python version: {platform.python_version()}")

    # Check PyTorch and CUDA
    try:
        import torch
        env_info["pytorch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda
            env_info["gpu_name"] = torch.cuda.get_device_name(0)
            env_info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            total_vram = (getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)) / (1024**3)
            env_info["total_vram_gb"] = round(total_vram, 2)
            print(f"[OK] PyTorch {torch.__version__} with CUDA {torch.version.cuda}")
            print(f"[OK] GPU: {env_info['gpu_name']} ({env_info['total_vram_gb']} GB VRAM)")
            print(f"[OK] GPU count: {env_info['gpu_count']}")
        else:
            errors.append("CUDA is not available")
            print("[FAIL] CUDA is not available")
    except ImportError:
        errors.append("PyTorch not installed")
        print("[FAIL] PyTorch not installed")

    # Check critical imports
    critical_libs = {
        "transformers": "transformers",
        "peft": "peft",
        "trl": "trl",
        "bitsandbytes": "bitsandbytes",
        "fastapi": "fastapi",
        "gradio": "gradio",
        "accelerate": "accelerate",
        "datasets": "datasets",
        "huggingface_hub": "huggingface_hub",
        "PIL": "Pillow",
        "qwen_vl_utils": "qwen-vl-utils",
    }

    installed_versions = {}
    for module_name, package_name in critical_libs.items():
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            installed_versions[package_name] = version
            print(f"[OK] {package_name}: {version}")
        except ImportError as e:
            errors.append(f"{package_name} not installed: {e}")
            print(f"[FAIL] {package_name}: {e}")

    env_info["installed_packages"] = installed_versions

    # Summary
    print(f"\n{'='*50}")
    if errors:
        print(f"VALIDATION FAILED — {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
    else:
        print("ALL VALIDATIONS PASSED")
    print(f"{'='*50}")

    env_info["validation_passed"] = len(errors) == 0
    env_info["errors"] = errors
    return env_info


if __name__ == "__main__":
    env_info = validate_environment()

    # Save to benchmarks/environment.json
    benchmarks_dir = Path(__file__).parent.parent / "benchmarks"
    benchmarks_dir.mkdir(exist_ok=True)
    output_path = benchmarks_dir / "environment.json"
    with open(output_path, "w") as f:
        json.dump(env_info, f, indent=2)
    print(f"\nEnvironment info saved to {output_path}")

    sys.exit(0 if env_info["validation_passed"] else 1)
