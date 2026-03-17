import os
import re
import shutil
import argparse
from pathlib import Path

IMG_RE = re.compile(r"^img_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
FLOW_RE = re.compile(r"^flow_(\d+)\.png$", re.IGNORECASE)

def list_indexed_files(dir_path: Path, pattern_re: re.Pattern):
    items = []
    if not dir_path.exists():
        return items
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        m = pattern_re.match(p.name)
        if m:
            idx = int(m.group(1))
            items.append((idx, p))
    items.sort(key=lambda x: x[0])
    return items

def safe_rename_dir(src: Path, dst: Path, dry_run: bool):
    if src == dst:
        return dst
    if dst.exists():
        # 已存在同名目标，直接跳过，避免覆盖
        print(f"[SKIP] DIR EXISTS: {dst}")
        return src
    print(f"[{'DRY' if dry_run else 'DO '}] DIRRENAME {src} -> {dst}")
    if not dry_run:
        src.rename(dst)
    return dst

def pad_flow(flow_dir: Path, target_count: int, dry_run: bool):
    flow_list = list_indexed_files(flow_dir, FLOW_RE)
    cur = len(flow_list)
    if cur == 0:
        print(f"[WARN] EMPTY FLOW: {flow_dir}")
        return

    if cur == target_count:
        return

    # 用最后一帧补齐
    last_idx, last_path = flow_list[-1]

    if cur > target_count:
        # 不建议自动删，先提示
        print(f"[WARN] FLOW > RGB ({cur} > {target_count}) : {flow_dir} (no delete)")
        return

    # cur < target_count: 需要补齐
    # 目标编号通常是从 1 到 target_count；也兼容不连续
    # 我们按 “当前最大idx + 1 ...” 或 “target_count” 生成
    # 更稳妥：如果 idx 从1开始连续，则新文件名用 target_count 对应编号
    # 这里按 next_idx 递增补齐到 target_count
    next_idx = last_idx + 1
    while cur < target_count:
        new_name = f"flow_{next_idx:05d}.png"
        dst = flow_dir / new_name
        if dst.exists():
            # 目标已存在，向后挪一位继续
            next_idx += 1
            continue

        print(f"[{'DRY' if dry_run else 'DO '}] PAD {last_path.name} -> {dst.name}  ({flow_dir})")
        if not dry_run:
            shutil.copy2(last_path, dst)

        cur += 1
        next_idx += 1

def main():
    parser = argparse.ArgumentParser(
        description="Plan1: unify FlowRGB dir names with RGB_ prefix and pad flow frames to match RGB frame count."
    )
    parser.add_argument("--root", type=str, default="E:\\transformer实验代码\\DSCMT-main\\data",
                        help="Dataset root containing RGB/ and FlowRGB/ (default: DSCMT-main/data)")
    parser.add_argument("--dry_run", action="store_true", help="Only print operations, do not modify anything.")
    args = parser.parse_args()

    root = Path(args.root)
    rgb_root = root / "RGB"
    flow_root = root / "FlowRGB"

    if not rgb_root.exists():
        raise FileNotFoundError(f"RGB root not found: {rgb_root}")
    if not flow_root.exists():
        raise FileNotFoundError(f"FlowRGB root not found: {flow_root}")

    print(f"=== Plan1 Fix ===")
    print(f"ROOT   : {root}")
    print(f"RGB    : {rgb_root}")
    print(f"FlowRGB: {flow_root}")
    print(f"MODE   : {'DRY_RUN' if args.dry_run else 'APPLY'}")
    print()

    # 遍历 FlowRGB/<Action>/<SampleDir>
    for action_dir in sorted(flow_root.iterdir()):
        if not action_dir.is_dir():
            continue

        action_name = action_dir.name
        rgb_action_dir = rgb_root / action_name

        if not rgb_action_dir.exists():
            print(f"[WARN] RGB action folder missing for Flow action '{action_name}': {rgb_action_dir}")
            continue

        for sample_dir in sorted(action_dir.iterdir()):
            if not sample_dir.is_dir():
                continue

            src_flow_dir = sample_dir
            sample_name = src_flow_dir.name

            # 1) FlowRGB 样本目录名统一加 RGB_ 前缀
            #   例如 Charge_gjb_01_0 -> RGB_Charge_gjb_01_0
            if not sample_name.startswith("RGB_"):
                dst_flow_dir = action_dir / ("RGB_" + sample_name)
                flow_dir = safe_rename_dir(src_flow_dir, dst_flow_dir, args.dry_run)
            else:
                flow_dir = src_flow_dir

            # 2) 找对应的 RGB 样本目录：RGB/<Action>/RGB_<SampleNameWithoutRGB_?>
            #    规则：Flow里最终是 RGB_xxx，则 RGB 里也是 RGB_xxx
            rgb_sample_dir = rgb_action_dir / flow_dir.name
            if not rgb_sample_dir.exists():
                # 兼容：如果你的 RGB 目录是 RGB_RGB_xxx（极少见）或其它情况，给提示
                print(f"[WARN] RGB sample folder not found for flow sample: {flow_dir}")
                print(f"       expected: {rgb_sample_dir}")
                continue

            rgb_imgs = list_indexed_files(rgb_sample_dir, IMG_RE)
            if len(rgb_imgs) == 0:
                print(f"[WARN] EMPTY RGB: {rgb_sample_dir}")
                continue

            target_count = len(rgb_imgs)

            # 3) Flow 帧补齐到 target_count
            pad_flow(flow_dir, target_count, args.dry_run)

    print("\n✅ Done.")
    if args.dry_run:
        print("你刚刚是 dry_run 预演模式；确认输出没问题后，去掉 --dry_run 再跑一次正式修改。")

if __name__ == "__main__":
    main()
