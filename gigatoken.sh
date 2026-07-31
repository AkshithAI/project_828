#!/usr/bin/env bash
set -e

echo "=================================================="
echo " Building Gigatoken from Source for Peak Hardware "
echo "=================================================="

# 1. Verify / Install Rust Toolchain
if ! command -v cargo &> /dev/null; then
    echo "[+] Rust not found. Installing Rust toolchain via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "[✓] Rust compiler found: $(rustc --version)"
fi

# 2. Clone the Official Repository
REPO_DIR="gigatoken"
if [ -d "$REPO_DIR" ]; then
    echo "[!] Existing directory '$REPO_DIR' found. Updating repository..."
    cd "$REPO_DIR"
    git pull origin main
else
    echo "[+] Cloning https://github.com/marcelroed/gigatoken.git..."
    git clone https://github.com/marcelroed/gigatoken.git
    cd "$REPO_DIR"
fi

# 3. Set Hardware-Tuned Native CPU Compilation Flags
# RUSTFLAGS="-C target-cpu=native" forces rustc to compile SIMD extensions
# (AVX-512, AVX2, or ARM NEON) specific to your host machine.
export RUSTFLAGS="-C target-cpu=native"

echo "[+] Target CPU RUSTFLAGS set to native microarchitecture."

# 4. Upgrade Build Tools & Install
echo "[+] Building and installing wheel into active Python environment..."
pip install --upgrade pip setuptools maturin

# Build with --no-build-isolation to pass environment RUSTFLAGS to Maturin/Cargo
pip install --no-build-isolation -e .

# 5. Sanity Check
echo "=================================================="
echo "[✓] Verifying Installation..."
python3 -c "
import gigatoken as gt
print('Successfully imported Gigatoken!')
print('Installed path:', gt.__file__)
"
echo "=================================================="