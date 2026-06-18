#!/usr/bin/env bash
set -euo pipefail

# One-shot ARM64 + MPS environment bootstrap for Apple Silicon.
# Usage:
#   bash scripts/setup_arm64_mps_env.sh [VENV_PATH]
# Example:
#   bash scripts/setup_arm64_mps_env.sh "$HOME/venvs/pinn-arm64"

VENV_PATH="${1:-$HOME/venvs/pinn-arm64}"
PY_FORMULA="python@3.11"
BREW_ARM="/opt/homebrew/bin/brew"
BREW_INIT='eval "$(/opt/homebrew/bin/brew shellenv)"'

say() { printf "[setup-arm64] %s\n" "$*"; }
fail() { printf "[setup-arm64] ERROR: %s\n" "$*" >&2; exit 1; }

say "Starting ARM64 + MPS setup"

HOST_ARCH="$(uname -m)"
if [[ "$HOST_ARCH" != "arm64" ]]; then
  fail "This script must run on Apple Silicon arm64. Current uname -m: $HOST_ARCH"
fi

# Ensure we are not running in a Rosetta-translated shell.
SHELL_ARCH="$(arch)"
if [[ "$SHELL_ARCH" != "arm64" ]]; then
  fail "Shell arch is $SHELL_ARCH. Open a native terminal (not Rosetta) and rerun."
fi

# Install ARM Homebrew if missing.
if [[ ! -x "$BREW_ARM" ]]; then
  say "ARM Homebrew not found at /opt/homebrew. Installing Homebrew for arm64..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
  say "Found ARM Homebrew: $BREW_ARM"
fi

# Initialize ARM Homebrew for current shell.
# shellcheck disable=SC1091
if ! eval "$($BREW_ARM shellenv)"; then
  fail "Failed to initialize ARM Homebrew shell environment"
fi

# Persist ARM Homebrew initialization.
if [[ -f "$HOME/.zprofile" ]]; then
  if ! grep -Fq "$BREW_INIT" "$HOME/.zprofile"; then
    say "Adding ARM Homebrew init to ~/.zprofile"
    printf "\n%s\n" "$BREW_INIT" >> "$HOME/.zprofile"
  fi
else
  say "Creating ~/.zprofile with ARM Homebrew init"
  printf "%s\n" "$BREW_INIT" > "$HOME/.zprofile"
fi

say "Updating Homebrew metadata"
$BREW_ARM update

say "Installing $PY_FORMULA"
$BREW_ARM install "$PY_FORMULA"

PY_BIN="$($BREW_ARM --prefix)/opt/$PY_FORMULA/bin/python3.11"
if [[ ! -x "$PY_BIN" ]]; then
  fail "Could not locate ARM Python at expected path: $PY_BIN"
fi

say "Using Python: $PY_BIN"
"$PY_BIN" -c 'import platform,sys; print("python_machine=", platform.machine()); print("python_executable=", sys.executable)'

say "Creating virtual environment at: $VENV_PATH"
"$PY_BIN" -m venv "$VENV_PATH"
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

say "Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

say "Installing core packages (PyTorch, NumPy, Matplotlib)"
pip install --upgrade numpy matplotlib torch torchvision torchaudio

say "Verifying MPS backend"
python - <<'PY'
import platform
import torch
print("platform_machine=", platform.machine())
print("torch_version=", torch.__version__)
print("mps_built=", torch.backends.mps.is_built())
print("mps_available=", torch.backends.mps.is_available())
print("cuda_available=", torch.cuda.is_available())
if not torch.backends.mps.is_available():
    print("WARNING: MPS is not available. Check: native arm64 shell, macOS version, and PyTorch build.")
PY

say "Setup complete"
say "Activate with: source \"$VENV_PATH/bin/activate\""
say "Recommended run prefix: PYTORCH_ENABLE_MPS_FALLBACK=1"
