#!/usr/bin/env bash
# Build the Kaggle Dataset that carries the VR M3.1 robot to a notebook.
#
# The M3.1 asset and motions live under data/assets/vr_m3_1 and
# data/motions/vr_m3_1, and both directories are gitignored (data/assets/.gitignore
# and data/motions/.gitignore are a bare `*`). That is the right call for 97 MB of
# STL and pickles, but it means a `git clone` on Kaggle gets the configs and none
# of the data they point at. A Kaggle Dataset is the transport.
#
#   bash kaggle/make_m3_dataset.sh                    # zombie_walk + spinkick, ~70 MB
#   bash kaggle/make_m3_dataset.sh --all-motions      # all 266 clips, ~97 MB
#   bash kaggle/make_m3_dataset.sh --clip vr_m3_1_humanoid_cartwheel.pkl
#
# The layout it writes is what kaggle/prepare_data.py recognises as a data pack
# (a directory holding both assets/ and motions/), so on the notebook side it is
# one extra line:
#
#   !python kaggle/prepare_data.py                              # the 534 MB pack
#   !python kaggle/prepare_data.py --src /kaggle/input/<slug>   # this one
#
# Two calls are needed because find_source() returns the first pack it finds and
# stops. link_tree() merges and leaves existing entries alone, so the second call
# only adds vr_m3_1 without disturbing the first.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/kaggle_dataset_vr_m3_1"
# zombie_walk first: it ranked 2nd of 267 clips under `retime_motion.py --scan`
# (no flight phase, worst leg torque 0.28x) and is the recommended first motion for
# this robot. spinkick rides along for comparison, but see SMP_PLAYBOOK.md 4.1 --
# it needs a 27 cm jump and cost a 320M-sample run.
CLIPS=("vr_m3_1_humanoid_zombie_walk.pkl" "vr_m3_1_humanoid_spinkick.pkl")
ALL_MOTIONS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all-motions) ALL_MOTIONS=1; shift ;;
        --clip)        CLIPS+=("$2"); shift 2 ;;
        --out)         OUT_DIR="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

SRC_ASSETS="${REPO_ROOT}/data/assets/vr_m3_1"
SRC_MOTIONS="${REPO_ROOT}/data/motions/vr_m3_1"

for d in "$SRC_ASSETS" "$SRC_MOTIONS"; do
    if [[ ! -d "$d" ]]; then
        echo "ERROR: missing $d" >&2
        echo "Copy the robot in from the vinrobotics_mjlab checkout first." >&2
        exit 1
    fi
done

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/assets" "$OUT_DIR/motions/vr_m3_1"

# The whole asset directory always goes: MuJoCo needs every mesh the MJCF
# references, both to build the model and to render it.
cp -r "$SRC_ASSETS" "$OUT_DIR/assets/vr_m3_1"

if [[ $ALL_MOTIONS -eq 1 ]]; then
    cp "$SRC_MOTIONS"/*.pkl "$OUT_DIR/motions/vr_m3_1/"
else
    for clip in "${CLIPS[@]}"; do
        if [[ ! -f "$SRC_MOTIONS/$clip" ]]; then
            echo "ERROR: no such clip: $SRC_MOTIONS/$clip" >&2
            exit 1
        fi
        cp "$SRC_MOTIONS/$clip" "$OUT_DIR/motions/vr_m3_1/"
    done
fi

cat > "$OUT_DIR/dataset-metadata.json" <<'JSON'
{
  "title": "MimicKit VR M3.1",
  "id": "REPLACE_WITH_YOUR_USERNAME/mimickit-vr-m3-1",
  "licenses": [{"name": "other"}]
}
JSON

echo "=== $OUT_DIR ==="
echo "  assets : $(find "$OUT_DIR/assets" -type f | wc -l) file(s)"
echo "  motions: $(find "$OUT_DIR/motions" -type f -name '*.pkl' | wc -l) clip(s)"
echo "  total  : $(du -sh "$OUT_DIR" | cut -f1)"
echo
echo "Next:"
echo "  1. edit ${OUT_DIR}/dataset-metadata.json -- put your Kaggle username in \"id\""
echo "  2. kaggle datasets create -p ${OUT_DIR} --dir-mode zip"
echo "     (or upload the folder by hand at kaggle.com/datasets/new)"
echo "  3. attach it to the notebook alongside the main MimicKit data pack"
