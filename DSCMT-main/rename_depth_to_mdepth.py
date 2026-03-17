import os
import re
import argparse

IMG_RE = re.compile(r"^img_(\d{5})\.png$", re.IGNORECASE)

def rename_one_folder(folder: str, commit: bool):
    files = os.listdir(folder)
    hits = []
    for fn in files:
        m = IMG_RE.match(fn)
        if m:
            idx5 = int(m.group(1))              # 1..99999
            src = os.path.join(folder, fn)
            dst = os.path.join(folder, f"MDepth-{idx5:08d}.png")
            hits.append((idx5, src, dst))

    if not hits:
        return 0

    # 按序号排序，避免乱序
    hits.sort(key=lambda x: x[0])

    # 先检查是否会覆盖已有文件
    for _, _, dst in hits:
        if os.path.exists(dst):
            raise RuntimeError(f"[冲突] 目标文件已存在，停止：{dst}")

    # 执行改名
    for _, src, dst in hits:
        if commit:
            os.rename(src, dst)
        else:
            print("[DRYRUN]", os.path.basename(src), "->", os.path.basename(dst))

    return len(hits)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Depth 根目录，例如 E:\\transformer实验代码\\DSCMT-main\\data\\Depth")
    ap.add_argument("--commit", action="store_true", help="真正执行改名（不加则只预览）")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise FileNotFoundError(f"root 不存在：{root}")

    total = 0
    # 遍历：Depth/<class>/<video_folder>
    for cls in os.listdir(root):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue

        for vid_folder in os.listdir(cls_path):
            vid_path = os.path.join(cls_path, vid_folder)
            if not os.path.isdir(vid_path):
                continue

            # 只处理 D_ 开头的序列文件夹（匹配你的结构）
            if not vid_folder.startswith("D_"):
                continue

            n = rename_one_folder(vid_path, args.commit)
            if n > 0:
                print(f"[OK] {cls}/{vid_folder}: {n} files")
                total += n

    print(f"\nDone. renamed = {total} files. commit={args.commit}")

if __name__ == "__main__":
    main()
