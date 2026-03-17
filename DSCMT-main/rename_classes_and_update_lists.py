import os
import re
import argparse
from pathlib import Path

def build_mapping_from_existing_classes(*roots: Path):
    """
    扫描 data/RGB, data/Depth, data/FlowRGB 下的“类别文件夹名”
    找出包含空格的类别，并生成: 'Wipe up the water' -> 'Wipe_up_the_water'
    """
    classes = set()
    for r in roots:
        if not r.exists():
            continue
        for p in r.iterdir():
            if p.is_dir():
                classes.add(p.name)

    mapping = {}
    for c in sorted(classes):
        if " " in c:
            mapping[c] = c.replace(" ", "_")
    return mapping

def safe_rename_dir(src: Path, dst: Path, commit: bool):
    if not src.exists():
        return
    if dst.exists():
        # 已经有目标目录了，避免覆盖；这里选择跳过并提示
        print(f"[SKIP] dst exists: {dst}")
        return
    if commit:
        src.rename(dst)
        print(f"[RENAME] {src} -> {dst}")
    else:
        print(f"[DRYRUN] {src} -> {dst}")

def update_list_file(list_path: Path, mapping: dict, commit: bool):
    """
    list 格式：<rel_path> <num_frames> <label>
    其中 rel_path 可能包含空格（就是你现在炸掉的原因）
    我们从右往左解析，最后两列一定是 num_frames 和 label，前面全部拼回 rel_path。
    然后仅替换 rel_path 的“类别名部分”（斜杠前的 class）。
    """
    if not list_path.exists():
        print(f"[SKIP] list not found: {list_path}")
        return

    new_lines = []
    changed = False

    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                new_lines.append(raw)
                continue

            parts = raw.split()
            if len(parts) < 3:
                # 异常行，原样保留
                new_lines.append(raw)
                continue

            label = parts[-1]
            numf = parts[-2]
            rel_path = " ".join(parts[:-2])  # 关键：把前面重新拼回去

            # rel_path 期望为 "Class/VideoId"
            rel_path_norm = rel_path.replace("\\", "/")
            if "/" in rel_path_norm:
                cls, rest = rel_path_norm.split("/", 1)
                if cls in mapping:
                    cls2 = mapping[cls]
                    rel_path_norm2 = cls2 + "/" + rest
                    if rel_path_norm2 != rel_path_norm:
                        changed = True
                    rel_path_norm = rel_path_norm2

            new_line = f"{rel_path_norm} {numf} {label}"
            new_lines.append(new_line)

    if not changed:
        print(f"[OK] {list_path.name}: no change")
        return

    if commit:
        list_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        print(f"[WRITE] {list_path.name}: updated")
    else:
        print(f"[DRYRUN] {list_path.name}: would update")

def main():
    ap = argparse.ArgumentParser(
        description="Rename class folders (space -> underscore) and update train/val/test list files accordingly."
    )
    ap.add_argument("--project_root", type=str, default=str(Path(__file__).resolve().parent),
                    help="DSCMT-main project root (default: script directory)")
    ap.add_argument("--commit", action="store_true",
                    help="Actually perform renaming and overwrite list files. Default is dry-run.")
    ap.add_argument("--rgb_dir", type=str, default="data/RGB")
    ap.add_argument("--depth_dir", type=str, default="data/Depth")
    ap.add_argument("--flow_dir", type=str, default="data/FlowRGB")
    ap.add_argument("--lists", nargs="*", default=[
        "train_test_files/train_universal.txt",
        "train_test_files/val_universal.txt",
        "train_test_files/test_universal.txt",
    ], help="List files to update")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    rgb_root = project_root / args.rgb_dir
    depth_root = project_root / args.depth_dir
    flow_root = project_root / args.flow_dir

    mapping = build_mapping_from_existing_classes(rgb_root, depth_root, flow_root)

    print("=== Class mapping ===")
    if not mapping:
        print("(no class with spaces found)")
    else:
        for k, v in mapping.items():
            print(f"{k} -> {v}")
    print("=====================")

    # 1) rename class folders under RGB/Depth/FlowRGB
    for base in (rgb_root, depth_root, flow_root):
        if not base.exists():
            continue
        for old, new in mapping.items():
            src = base / old
            dst = base / new
            safe_rename_dir(src, dst, commit=args.commit)

    # 2) update list files
    for lp in args.lists:
        update_list_file(project_root / lp, mapping, commit=args.commit)

    print(f"\nDone. commit={args.commit}")

if __name__ == "__main__":
    main()
