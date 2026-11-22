#!/usr/bin/env bash
# Guided install of gpu-unlock (https://github.com/Zagitta/gpu-unlock), GPL-2.0, by its author's own steps.
#
# It is a Linux kernel module that runs at boot, before the NVIDIA driver, and lifts the GeForce half-rate cap on
# FP16/BF16 tensor math with FP32 accumulation (RTX 30 stable, RTX 40/20 experimental, RTX 50 unsupported). It needs
# root, builds against your kernel, and only acts when the kernel command line opts in with gpu_unlock.targets=...
#
# This script never edits your bootloader: it prints the exact line to add, so the change stays in your hands.
# Undo: remove the kernel option (or uninstall the package) and reboot.
set -euo pipefail

REPO=https://github.com/Zagitta/gpu-unlock
REF=${GPU_UNLOCK_REF:-}  # optionally pin a tag/commit you have reviewed
DIR=${GPU_UNLOCK_DIR:-$HOME/.cache/tinyh3/gpu-unlock}

say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }
ask() { read -r -p "$1 [y/N] " reply; [[ "$reply" =~ ^[YyДд]$ ]]; }

[[ "$(uname -s)" == Linux ]] || { warn "gpu-unlock is Linux-only."; exit 1; }
command -v nvidia-smi >/dev/null || { warn "nvidia-smi not found: install the NVIDIA driver first."; exit 1; }

say "GPUs:"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -Eqi 'RTX (20|30|40)'; then
  warn "No RTX 20/30/40 GeForce found; gpu-unlock would not change anything here."
  exit 1
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Eqi 'RTX (20|40)'; then
  warn "Support for RTX 20/40 is marked experimental by the gpu-unlock author."
fi

cat <<EOF

This will:
  1. clone $REPO into $DIR${REF:+ (ref $REF)}
  2. run its 'make install' (builds the DKMS package for your distribution; asks for sudo)
  3. print the kernel option to add; nothing changes until you add it and reboot.

Read the project README first: $REPO
EOF
ask "Continue?" || exit 0

if [[ -d "$DIR/.git" ]]; then
  git -C "$DIR" fetch --tags --quiet
else
  git clone --quiet "$REPO" "$DIR"
fi
[[ -n "$REF" ]] && git -C "$DIR" checkout --quiet "$REF"
say "gpu-unlock $(cat "$DIR/VERSION" 2>/dev/null || echo '?') at $(git -C "$DIR" rev-parse --short HEAD)"

ask "Build and install it now (sudo)?" || exit 0
make -C "$DIR" install

cat <<'EOF'

Installed. To enable it, append this to your kernel command line, keeping the existing options:

    gpu_unlock.targets=all

  GRUB (Debian/Ubuntu/Mint): edit GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub, then: sudo update-grub
  GRUB (Arch):               same edit, then: sudo grub-mkconfig -o /boot/grub/grub.cfg
  Fedora/RHEL:               sudo grubby --update-kernel=ALL --args="gpu_unlock.targets=all"
  systemd-boot:              append to the 'options' line of your boot entry

After rebooting, check it with:

    tinyh3 doctor --refresh

TinyH3 measures the tensor rates and switches its attention kernels automatically.
EOF
