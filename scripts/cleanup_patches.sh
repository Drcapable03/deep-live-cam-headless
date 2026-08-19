#!/bin/bash
# ============================================================================
# Deep-Live-Cam Headless Streaming — Cleanup Patch Scripts
# ============================================================================
# Removes temporary ad-hoc fix scripts from the project root after confirming
# the codebase is stable (Phases 1-4 completed, all fixes merged).
#
# Usage:
#   chmod +x scripts/cleanup_patches.sh
#   ./scripts/cleanup_patches.sh
#
# Safe to re-run — only removes files that exist.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================================"
echo " Deep-Live-Cam — Cleanup Ad-Hoc Fix Scripts"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

REMOVED_COUNT=0
SKIPPED_COUNT=0

cleanup_file() {
    local filepath="${PROJECT_ROOT}/$1"
    if [ -f "${filepath}" ]; then
        echo "Removing: $1"
        rm -f "${filepath}"
        REMOVED_COUNT=$((REMOVED_COUNT + 1))
    else
        echo "Skipped (not found): $1"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    fi
}

echo "The following ad-hoc fix scripts will be removed:"
echo ""

for script in \
    "2fixed_py.sh" \
    "526_enhance.sh" \
    "correct_corefile.sh" \
    "ffmpeg_fix.sh" \
    "finxed_py.sh" \
    "just_fix.sh"; do
    echo "  ${script}"
done

echo ""

# Confirm before removal
read -p "Proceed with removal? [y/N]: " confirm
if [[ ! "${confirm:-N}" =~ ^[Yy]$ ]]; then
    echo "Aborted. No files were removed."
    exit 0
fi

echo ""
echo "--- Removing ---"

# Remove each script
cleanup_file "2fixed_py.sh"
cleanup_file "526_enhance.sh"
cleanup_file "correct_corefile.sh"
cleanup_file "ffmpeg_fix.sh"
cleanup_file "finxed_py.sh"
cleanup_file "just_fix.sh"

echo ""
echo "============================================================"
echo " Cleanup Complete!"
echo " Removed:     ${REMOVED_COUNT} file(s)"
echo " Skipped:     ${SKIPPED_COUNT} file(s) (not found)"
echo "============================================================"
echo ""
echo "All remaining code fixes are now integrated into the"
echo "codebase via Phases 1-4 development commits."
echo ""
