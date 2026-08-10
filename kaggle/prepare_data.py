"""Link the MimicKit asset/motion/model pack into `data/` on Kaggle.

The 534 MB data pack is not in git. Upload it once as a Kaggle Dataset (see
docs/README_Kaggle.md), then run this script: it finds the pack under
/kaggle/input and symlinks its contents into the repo's `data/` directory,
leaving files already tracked in git untouched.

Usage:
    python kaggle/prepare_data.py                 # auto-detect under /kaggle/input
    python kaggle/prepare_data.py --src <path>    # explicit source directory
"""

import argparse
import os
import sys

# Top-level directories the data pack is expected to provide.
DATA_DIRS = ["assets", "motions", "models", "logs"]


def find_source(search_root):
    """Return the directory that holds assets/ and motions/, or None."""
    if not os.path.isdir(search_root):
        return None

    # Look at the dataset roots and one level below (the pack is often nested
    # inside a MimicKit_Data/ folder).
    candidates = [search_root]
    for entry in sorted(os.listdir(search_root)):
        path = os.path.join(search_root, entry)
        if not os.path.isdir(path):
            continue
        candidates.append(path)
        for sub in sorted(os.listdir(path)):
            sub_path = os.path.join(path, sub)
            if os.path.isdir(sub_path):
                candidates.append(sub_path)

    for cand in candidates:
        has_assets = os.path.isdir(os.path.join(cand, "assets"))
        has_motions = os.path.isdir(os.path.join(cand, "motions"))
        if has_assets and has_motions:
            return cand
    return None


def link_tree(src, dst, stats):
    """Recursively mirror `src` into `dst` using symlinks for leaf entries."""
    os.makedirs(dst, exist_ok=True)

    for name in sorted(os.listdir(src)):
        src_path = os.path.join(src, name)
        dst_path = os.path.join(dst, name)

        if os.path.isdir(src_path) and os.path.isdir(dst_path) and not os.path.islink(dst_path):
            link_tree(src_path, dst_path, stats)
            continue

        if os.path.lexists(dst_path):
            # A stale symlink from a previous run gets refreshed; a real file
            # (i.e. one tracked in git) is kept as-is.
            if os.path.islink(dst_path):
                os.unlink(dst_path)
            else:
                stats["skipped"] += 1
                continue

        os.symlink(os.path.abspath(src_path), dst_path)
        stats["linked"] += 1
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None, help="directory containing assets/, motions/, models/")
    parser.add_argument("--input_root", default="/kaggle/input", help="where to search when --src is omitted")
    parser.add_argument("--data_dir", default="data", help="repo data directory to populate")
    args = parser.parse_args()

    src = args.src or find_source(args.input_root)
    if src is None:
        print("ERROR: could not find the MimicKit data pack under {}".format(args.input_root))
        print("Attach the dataset to the notebook, or pass --src <path>.")
        return 1
    print("Data pack: {}".format(src))

    stats = {"linked": 0, "skipped": 0}
    for name in DATA_DIRS:
        sub_src = os.path.join(src, name)
        if not os.path.isdir(sub_src):
            print("  - {:<8} (missing, skipped)".format(name))
            continue
        link_tree(sub_src, os.path.join(args.data_dir, name), stats)
        print("  + {:<8} -> {}".format(name, os.path.join(args.data_dir, name)))

    print("Linked {} entries, kept {} existing files.".format(stats["linked"], stats["skipped"]))

    # Sanity check on the two files the humanoid examples need.
    for check in ["assets/humanoid/humanoid.xml",
                  "motions/humanoid/humanoid_spinkick.pkl"]:
        path = os.path.join(args.data_dir, check)
        status = "ok" if os.path.exists(path) else "MISSING"
        print("  [{}] {}".format(status, path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
