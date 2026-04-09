#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  MHSF Triangle Detection – Jetson Orin Nano Setup Script
# ═══════════════════════════════════════════════════════════════
#  This script:
#    1. Detects the NVIDIA JetPack version
#    2. Installs Python dependencies from requirements.txt
#    3. Verifies every import succeeds
#    4. Creates a systemd service for auto-start on boot
#
#  Usage:
#    chmod +x setup.sh
#    sudo ./setup.sh
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

# ── Colours for pretty output ───────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Colour

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[  OK]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $*"; }

# ── Must run as root ────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    fail "This script must be run with: sudo ./setup.sh"
    exit 1
fi

# ── Resolve project directory (where this script lives) ────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR"
APP_MAIN="$APP_DIR/main.py"
REQ_FILE="$APP_DIR/requirements.txt"

# Detect the real (non-root) user who invoked sudo
REAL_USER="${SUDO_USER:-$(logname 2>/dev/null || echo $USER)}"
REAL_HOME="$(eval echo "~$REAL_USER")"

info "Project directory : $APP_DIR"
info "Running as user   : $REAL_USER"

# ═══════════════════════════════════════════════════════════════
#  STEP 1 – Detect JetPack Version
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  STEP 1 : Detecting JetPack Version"
echo "═══════════════════════════════════════════════════════════"

JETPACK_VERSION="unknown"
L4T_VERSION="unknown"

# Method 1: dpkg (JetPack meta-package)
if dpkg -l 2>/dev/null | grep -q "nvidia-jetpack"; then
    JETPACK_VERSION=$(dpkg -l | grep "nvidia-jetpack " | awk '{print $3}' | head -1)
    info "JetPack detected via dpkg: $JETPACK_VERSION"
fi

# Method 2: apt-cache
if [[ "$JETPACK_VERSION" == "unknown" ]]; then
    JP_APT=$(apt-cache show nvidia-jetpack 2>/dev/null | grep "^Version:" | awk '{print $2}' | head -1 || true)
    if [[ -n "$JP_APT" ]]; then
        JETPACK_VERSION="$JP_APT"
        info "JetPack detected via apt-cache: $JETPACK_VERSION"
    fi
fi

# Method 3: /etc/nv_tegra_release (L4T string)
if [[ -f /etc/nv_tegra_release ]]; then
    L4T_VERSION=$(head -1 /etc/nv_tegra_release | sed 's/.*R\([0-9]*\).*/R\1/' | head -c 10)
    L4T_FULL=$(head -1 /etc/nv_tegra_release)
    info "L4T release: $L4T_FULL"
fi

# Method 4: nv_jetson_release (Orin / newer JetPack 6.x)
if command -v nv_jetson_release &>/dev/null; then
    NV_OUT=$(nv_jetson_release 2>/dev/null || true)
    if [[ -n "$NV_OUT" ]]; then
        info "nv_jetson_release output:"
        echo "$NV_OUT" | while IFS= read -r line; do echo "    $line"; done
    fi
fi

# Method 5: jtop (if installed)
if [[ "$JETPACK_VERSION" == "unknown" ]] && command -v jtop &>/dev/null; then
    JP_JTOP=$(jtop --version 2>/dev/null | grep -i jetpack | awk '{print $NF}' || true)
    if [[ -n "$JP_JTOP" ]]; then
        JETPACK_VERSION="$JP_JTOP"
        info "JetPack detected via jtop: $JETPACK_VERSION"
    fi
fi

if [[ "$JETPACK_VERSION" == "unknown" ]]; then
    warn "Could not determine JetPack version. Continuing anyway..."
else
    success "JetPack version: $JETPACK_VERSION"
fi

# Detect CUDA
CUDA_VERSION="not found"
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    success "CUDA version: $CUDA_VERSION"
elif [[ -d /usr/local/cuda ]]; then
    CUDA_VERSION=$(cat /usr/local/cuda/version.txt 2>/dev/null | awk '{print $NF}' || echo "present")
    success "CUDA found at /usr/local/cuda ($CUDA_VERSION)"
else
    warn "CUDA not detected on PATH"
fi

# Detect Python
PYTHON_BIN=""
for py in python3 python; do
    if command -v "$py" &>/dev/null; then
        PYTHON_BIN="$(command -v "$py")"
        break
    fi
done
if [[ -z "$PYTHON_BIN" ]]; then
    fail "Python 3 not found. Install python3 first."
    exit 1
fi
PY_VERSION=$("$PYTHON_BIN" --version 2>&1)
success "Python: $PY_VERSION ($PYTHON_BIN)"

# ═══════════════════════════════════════════════════════════════
#  STEP 2 – Install Dependencies
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  STEP 2 : Installing Dependencies"
echo "═══════════════════════════════════════════════════════════"

# System packages
info "Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-keysyms1 \
    libxcb-shape0 \
    libdbus-1-3 \
    2>/dev/null || warn "Some system packages may have failed (non-critical)"
success "System packages installed"

# pip install (as the real user to avoid root site-packages issues)
if [[ -f "$REQ_FILE" ]]; then
    info "Installing Python packages from requirements.txt ..."
    sudo -u "$REAL_USER" "$PYTHON_BIN" -m pip install --upgrade pip 2>/dev/null || true
    sudo -u "$REAL_USER" "$PYTHON_BIN" -m pip install -r "$REQ_FILE" 2>&1 | tail -5
    success "pip install completed"
else
    warn "requirements.txt not found at $REQ_FILE — skipping pip install"
fi

# On JetPack, torch + torchvision are usually pre-installed via the
# NVIDIA wheel.  Warn if torch is missing so the user knows.
if ! sudo -u "$REAL_USER" "$PYTHON_BIN" -c "import torch" 2>/dev/null; then
    warn "PyTorch not found.  On Jetson, install the NVIDIA wheel:"
    warn "  https://forums.developer.nvidia.com/t/pytorch-for-jetson/"
    warn "  Example: pip install torch-<ver>-cp<ver>-linux_aarch64.whl"
fi

# ═══════════════════════════════════════════════════════════════
#  STEP 3 – Verify Installation
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  STEP 3 : Verifying Installation"
echo "═══════════════════════════════════════════════════════════"

VERIFY_PASS=0
VERIFY_FAIL=0

check_module() {
    local mod="$1"
    if sudo -u "$REAL_USER" "$PYTHON_BIN" -c "import $mod" 2>/dev/null; then
        local ver
        ver=$(sudo -u "$REAL_USER" "$PYTHON_BIN" -c "import $mod; print(getattr($mod, '__version__', 'ok'))" 2>/dev/null)
        success "$mod  ($ver)"
        ((VERIFY_PASS++)) || true
    else
        fail "$mod  — NOT INSTALLED"
        ((VERIFY_FAIL++)) || true
    fi
}

check_module cv2
check_module numpy
check_module PySide6
check_module ultralytics
check_module torch

echo ""
if [[ $VERIFY_FAIL -eq 0 ]]; then
    success "All modules verified ($VERIFY_PASS/$VERIFY_PASS)"
else
    warn "$VERIFY_FAIL module(s) missing.  The app may still run if optional."
fi

# Quick launch test (import only, no GUI)
info "Quick import test of main.py ..."
if sudo -u "$REAL_USER" "$PYTHON_BIN" -c "
import sys, os
sys.path.insert(0, '$APP_DIR')
os.chdir('$APP_DIR')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import main
print('main.py imported successfully')
" 2>/dev/null; then
    success "main.py import test passed"
else
    warn "main.py import test failed (non-fatal — may need display)"
fi

# ═══════════════════════════════════════════════════════════════
#  STEP 4 – Create Auto-Start Service
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  STEP 4 : Setting Up Auto-Start (systemd)"
echo "═══════════════════════════════════════════════════════════"

SERVICE_NAME="mhsf-triangle-detection"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

info "Creating systemd service: $SERVICE_NAME"

cat > "$SERVICE_FILE" << UNIT
[Unit]
Description=MHSF Triangle Detection – AI Inspection App
After=graphical.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$REAL_USER
WorkingDirectory=$APP_DIR
Environment=DISPLAY=:0
Environment=XAUTHORITY=$REAL_HOME/.Xauthority
Environment=XDG_RUNTIME_DIR=/run/user/$(id -u "$REAL_USER")
ExecStartPre=/bin/sleep 5
ExecStart=$PYTHON_BIN $APP_MAIN
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=graphical.target
UNIT

# Reload and enable
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
success "Service created and enabled: $SERVICE_FILE"

info "Service commands:"
echo "    Start now  :  sudo systemctl start  $SERVICE_NAME"
echo "    Stop       :  sudo systemctl stop   $SERVICE_NAME"
echo "    Status     :  sudo systemctl status $SERVICE_NAME"
echo "    Logs       :  journalctl -u $SERVICE_NAME -f"
echo "    Disable    :  sudo systemctl disable $SERVICE_NAME"

# ═══════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Setup Complete"
echo "═══════════════════════════════════════════════════════════"
success "JetPack     : $JETPACK_VERSION"
success "CUDA        : $CUDA_VERSION"
success "Python      : $PY_VERSION"
success "Modules OK  : $VERIFY_PASS   |   Missing : $VERIFY_FAIL"
success "Auto-start  : enabled ($SERVICE_NAME)"
echo ""
info "The app will auto-start after the next reboot."
info "To start it now:  sudo systemctl start $SERVICE_NAME"
echo ""
