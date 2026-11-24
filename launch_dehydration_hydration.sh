#!/usr/bin/env bash
# Launch the Dehydration/Hydration Correction GUI (mbirtorch hsnt, see
# README.md), rebuilding first if the sources changed. The same script serves
# production (rust_dehydration_hydration) and the beta (…_development).
#
# Usage: ./launch_dehydration_hydration.sh [dehydration_hydration arguments...]
#   e.g. ./launch_dehydration_hydration.sh /SNS/VENUS/IPTS-XXXX/.../Run_YYYY
#        ./launch_dehydration_hydration.sh --run-number 23640
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$REPO_DIR/target/release/dehydration_hydration"

# GUI apps need a display (e.g. a ThinLinc session).
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "Error: no display found (DISPLAY/WAYLAND_DISPLAY unset)." >&2
    echo "Run this from a graphical session such as ThinLinc." >&2
    exit 1
fi

# Rebuild if the binary is missing or any source/manifest file is newer.
needs_build=false
if [[ ! -x "$BINARY" ]]; then
    needs_build=true
elif [[ -n "$(find "$REPO_DIR/src" "$REPO_DIR/Cargo.toml" "$REPO_DIR/../rust_detector_orientation/src" "$REPO_DIR/../rust_session_replay/src" -newer "$BINARY" -print -quit 2>/dev/null)" ]]; then
    needs_build=true
fi

if $needs_build; then
    CARGO="$(command -v cargo || true)"
    [[ -z "$CARGO" && -x "$HOME/.cargo/bin/cargo" ]] && CARGO="$HOME/.cargo/bin/cargo"
    if [[ -z "$CARGO" ]]; then
        echo "Error: binary is out of date and cargo was not found to rebuild it." >&2
        exit 1
    fi
    echo "Building dehydration_hydration (release)..."
    # A build that fails (a shared crate being worked on…) must not keep the
    # tool from starting: the binary already there is launched.
    if ! (cd "$REPO_DIR" && "$CARGO" build --release); then
        if [[ -x "$BINARY" ]]; then
            echo "Warning: the build failed; launching the existing binary." >&2
        else
            echo "Error: the build failed and there is no binary to launch." >&2
            exit 1
        fi
    fi
fi

# Classic GTK file chooser (typeable path, honors the starting folder), not
# the XDG portal dialog.
export GTK_USE_PORTAL=0
exec "$BINARY" "$@"
