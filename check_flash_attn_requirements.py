#!/usr/bin/env python3
"""
Check system versions to determine the correct Flash Attention prebuilt wheel.

Flash Attention wheels are specific to:
- Python version (3.8, 3.9, 3.10, 3.11, 3.12)
- PyTorch version (2.0, 2.1, 2.2, 2.3, 2.4, 2.5)
- CUDA version (11.8, 12.1, 12.2, 12.3, 12.4)

Prebuilt wheels: https://github.com/Dao-AILab/flash-attention/releases
"""

import sys
import subprocess
import platform


def get_python_version():
    """Get Python version."""
    version = sys.version_info
    return f"{version.major}.{version.minor}.{version.micro}", f"{version.major}.{version.minor}"


def get_pytorch_version():
    """Get PyTorch version and CUDA version from PyTorch."""
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]  # Remove +cu118 suffix if present
        torch_major_minor = '.'.join(torch_version.split('.')[:2])
        
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        
        return {
            'torch_version': torch_version,
            'torch_major_minor': torch_major_minor,
            'cuda_available': cuda_available,
            'cuda_version': cuda_version,
            'cuda_arch_list': torch.cuda.get_arch_list() if cuda_available else [],
        }
    except ImportError:
        return None


def get_system_cuda_version():
    """Get system CUDA version from nvcc."""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            # Parse: "Cuda compilation tools, release 12.1, V12.1.105"
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    parts = line.split('release')[-1].strip()
                    version = parts.split(',')[0].strip()
                    return version
    except FileNotFoundError:
        pass
    return None


def get_gpu_info():
    """Get GPU information using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        gpus.append({
                            'name': parts[0],
                            'driver_version': parts[1],
                            'memory': parts[2]
                        })
            return gpus
    except FileNotFoundError:
        pass
    return []


def get_flash_attn_wheel_name(python_minor, torch_minor, cuda_version):
    """Generate the expected wheel filename pattern."""
    # Flash attention wheel naming convention:
    # flash_attn-{version}+cu{cuda}torch{torch}cxx11abiTRUE-cp{py}-cp{py}-linux_x86_64.whl
    cuda_short = cuda_version.replace('.', '') if cuda_version else 'xxx'
    torch_short = torch_minor.replace('.', '')
    
    return f"flash_attn-*+cu{cuda_short}torch{torch_short}*-cp{python_minor.replace('.', '')}-*-linux_x86_64.whl"


def check_flash_attn_installed():
    """Check if flash-attn is already installed."""
    try:
        import flash_attn
        return flash_attn.__version__
    except ImportError:
        return None


def main():
    print("=" * 60)
    print("Flash Attention Wheel Requirements Checker")
    print("=" * 60)
    
    # Python version
    py_full, py_minor = get_python_version()
    print(f"\n📌 Python Version: {py_full}")
    print(f"   Wheel tag: cp{py_minor.replace('.', '')}")
    
    # System info
    print(f"\n📌 Platform: {platform.system()} {platform.machine()}")
    
    # PyTorch info
    print("\n" + "-" * 60)
    torch_info = get_pytorch_version()
    if torch_info:
        print(f"🔥 PyTorch Version: {torch_info['torch_version']}")
        print(f"   Wheel tag: torch{torch_info['torch_major_minor'].replace('.', '')}")
        print(f"   CUDA Available: {torch_info['cuda_available']}")
        if torch_info['cuda_version']:
            print(f"   PyTorch CUDA Version: {torch_info['cuda_version']}")
            print(f"   Wheel tag: cu{torch_info['cuda_version'].replace('.', '')}")
        if torch_info['cuda_arch_list']:
            print(f"   Supported CUDA Architectures: {', '.join(torch_info['cuda_arch_list'])}")
    else:
        print("❌ PyTorch not installed!")
        print("   Install PyTorch first: pip install torch")
    
    # System CUDA
    print("\n" + "-" * 60)
    system_cuda = get_system_cuda_version()
    if system_cuda:
        print(f"🖥️  System CUDA (nvcc): {system_cuda}")
    else:
        print("⚠️  nvcc not found in PATH (may not be needed for prebuilt wheels)")
    
    # GPU info
    print("\n" + "-" * 60)
    gpus = get_gpu_info()
    if gpus:
        print("🎮 GPU(s) Detected:")
        for i, gpu in enumerate(gpus):
            print(f"   [{i}] {gpu['name']}")
            print(f"       Driver: {gpu['driver_version']}, Memory: {gpu['memory']}")
    else:
        print("❌ No NVIDIA GPU detected or nvidia-smi not available")
    
    # Flash attention status
    print("\n" + "-" * 60)
    flash_version = check_flash_attn_installed()
    if flash_version:
        print(f"✅ Flash Attention installed: {flash_version}")
    else:
        print("❌ Flash Attention not installed")
    
    # Recommendation
    print("\n" + "=" * 60)
    print("📦 RECOMMENDED WHEEL")
    print("=" * 60)
    
    if torch_info and torch_info['cuda_version']:
        cuda_v = torch_info['cuda_version']
        torch_v = torch_info['torch_major_minor']
        
        wheel_pattern = get_flash_attn_wheel_name(py_minor, torch_v, cuda_v)
        print(f"\nLook for wheel matching: {wheel_pattern}")
        
        print(f"\n🔗 Download from:")
        print(f"   https://github.com/Dao-AILab/flash-attention/releases")
        
        print(f"\n📋 Your specifications:")
        print(f"   • Python: {py_minor} (cp{py_minor.replace('.', '')})")
        print(f"   • PyTorch: {torch_v} (torch{torch_v.replace('.', '')})")
        print(f"   • CUDA: {cuda_v} (cu{cuda_v.replace('.', '')})")
        
        # Generate pip install command
        cuda_short = cuda_v.replace('.', '')
        torch_short = torch_v.replace('.', '')
        py_short = py_minor.replace('.', '')
        
        print(f"\n💡 Installation options:")
        print(f"\n   Option 1 - Build from source (slow but reliable):")
        print(f"   pip install flash-attn --no-build-isolation")
        
        print(f"\n   Option 2 - Direct wheel install (fast):")
        print(f"   # Download matching wheel from GitHub releases, then:")
        print(f"   pip install flash_attn-<version>+cu{cuda_short}torch{torch_short}cxx11abiTRUE-cp{py_short}-cp{py_short}-linux_x86_64.whl")
        
        print(f"\n   Option 3 - Using pip with URL (if exact version known):")
        print(f"   pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu{cuda_short}torch{torch_short}cxx11abiTRUE-cp{py_short}-cp{py_short}-linux_x86_64.whl")
        
    else:
        print("\n⚠️  Cannot determine wheel requirements.")
        print("   Make sure PyTorch with CUDA is installed first.")
        print("\n   Install PyTorch with CUDA:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
