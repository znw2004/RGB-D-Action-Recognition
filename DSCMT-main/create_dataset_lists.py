import os
import os.path as osp
import random

# =========================
# 配置区：已适配你的实际文件格式和Subject
# =========================
ROOT = r"E:\transformer实验代码\DSCMT-main"
RGB_ROOT = osp.join(ROOT, "data", "RGB")
DEP_ROOT = osp.join(ROOT, "data", "Depth")
FLOW_ROOT = osp.join(ROOT, "data", "FlowRGB")
OUT_DIR = osp.join(ROOT, "train_test_files")

# 适配你的文件夹前缀规则
RGB_PREFIX = "RGB_"  # RGB下是 RGB_Charge_gjb_01_0
DEP_PREFIX = "D_"  # Depth下是 D_Charge_gjb_01_0
FLOW_PREFIX = ""  # FlowRGB下直接是 Charge_gjb_01_0（无前缀）

# 帧文件名规则（按你的实际格式）
RGB_STEM = "img_"  # RGB帧：img_00001.jpg
DEP_STEM = "img_"  # Depth帧：img_00001.png

# Flow帧规则：单文件 flow_00001.png
FLOW_X_STEM = "flow_x_"
FLOW_Y_STEM = "flow_y_"
FLOW_1_STEM = "flow_"  # 匹配你的flow_00001.png

MIN_FRAMES = 5
SEED = 2026
random.seed(SEED)

# ===== A 方案：固定划分（适配你实际的6个Subject）=====
# 实际检测到的Subject：['gjb', 'lxy', 'lyh', 'xzl', 'yzf', 'znw']
# 按 5:1:0 划分（因只有6个，无法严格6:1:1）
TRAIN_SUBJECTS = ["gjb", "lxy", "lyh", "xzl", "yzf"]  # 5个训练
VAL_SUBJECTS = ["znw"]  # 1个验证
TEST_SUBJECTS = []  # 无测试样本（可根据新增数据调整）


# 如果后续补充了更多Subject，可恢复6:1:1划分，示例：
# TRAIN_SUBJECTS = ["gjb", "lxy", "lyh", "xzl", "yzf", "znw"]  # 6个训练
# VAL_SUBJECTS   = ["new_sub1"]                               # 1个验证
# TEST_SUBJECTS  = ["new_sub2"]                               # 1个测试


def list_dirs(p):
    if not osp.isdir(p):
        return []
    return sorted([d for d in os.listdir(p) if osp.isdir(osp.join(p, d))])


def count_rgb_frames(rgb_dir):
    # 统计 RGB 帧：img_00001.jpg
    c = 0
    for f in os.listdir(rgb_dir):
        lf = f.lower()
        if lf.startswith(RGB_STEM) and (lf.endswith(".jpg") or lf.endswith(".png")):
            c += 1
    return c


def count_depth_frames(dep_dir):
    # 统计 Depth 帧：img_00001.png（适配你的格式）
    c = 0
    for f in os.listdir(dep_dir):
        lf = f.lower()
        if lf.startswith(DEP_STEM) and lf.endswith(".png"):
            c += 1
    return c


def count_flow_frames(flow_dir):
    """
    返回 (mode, n):
      mode = "xy" | "single" | "none"
      n = 可用帧数（xy 用 min(fx,fy)，single 用 flow_ 数量）
    """
    fx = 0
    fy = 0
    single = 0

    for f in os.listdir(flow_dir):
        lf = f.lower()
        if lf.startswith(FLOW_X_STEM) and (lf.endswith(".jpg") or lf.endswith(".png")):
            fx += 1
        elif lf.startswith(FLOW_Y_STEM) and (lf.endswith(".jpg") or lf.endswith(".png")):
            fy += 1
        elif lf.startswith(FLOW_1_STEM) and (lf.endswith(".jpg") or lf.endswith(".png")):
            single += 1

    if fx > 0 and fy > 0:
        return "xy", min(fx, fy)
    if single > 0:
        return "single", single
    return "none", 0


def parse_subject(video_id_no_prefix):
    """
    适配两种格式：
    1. Charge_gjb_01_0（下划线分隔）
    2. Tie-lxy-01_1（混合连字符+下划线）
    提取其中的subject（比如 lxy/gjb）
    """
    # 先把连字符替换为下划线，统一处理
    normalized = video_id_no_prefix.replace("-", "_")
    parts = normalized.split("_")

    # 遍历所有部分，找到匹配的有效subject
    valid_subjects = {"gjb", "lxy", "lyh", "xzl", "yzf", "znw"}
    for part in parts:
        if part in valid_subjects:
            return part

    # 没找到则打印警告（不影响整体运行）
    print(f"[WARN] 无法解析subject：{video_id_no_prefix}（格式不符）")
    return None


def build_common_classes():
    rgb_cls = set(list_dirs(RGB_ROOT))
    dep_cls = set(list_dirs(DEP_ROOT))
    flo_cls = set(list_dirs(FLOW_ROOT))
    common = sorted(list(rgb_cls & dep_cls & flo_cls))
    return common


def build_samples(common_classes):
    """
    返回 samples: list of (rel_id, n_frames, label0, subject)
    其中 rel_id 用于写入 txt：  class_name/video_id_no_prefix
    label0 是 0-based 类别编号（按 common_classes 的排序）
    """
    samples = []
    bad_name = 0
    too_short = 0
    missing_triplet = 0

    # 类别 -> 0-based label
    class2label = {cls: i for i, cls in enumerate(common_classes)}

    for ci, cls in enumerate(common_classes, 1):
        rgb_cls_dir = osp.join(RGB_ROOT, cls)
        dep_cls_dir = osp.join(DEP_ROOT, cls)
        flo_cls_dir = osp.join(FLOW_ROOT, cls)

        # 遍历 RGB 类别下的所有 RGB_* 目录作为候选视频
        for d in list_dirs(rgb_cls_dir):
            if not d.startswith(RGB_PREFIX):
                continue

            vid = d[len(RGB_PREFIX):]  # 去掉 RGB_ 前缀，得到 Charge_gjb_01_0
            subject = parse_subject(vid)
            if subject is None:
                bad_name += 1
                continue

            # 拼接三模态的完整路径（适配Flow无前缀）
            rgb_dir = osp.join(rgb_cls_dir, RGB_PREFIX + vid)
            dep_dir = osp.join(dep_cls_dir, DEP_PREFIX + vid)
            flo_dir = osp.join(flo_cls_dir, FLOW_PREFIX + vid)

            # 检查三模态文件夹是否都存在
            if not (osp.isdir(rgb_dir) and osp.isdir(dep_dir) and osp.isdir(flo_dir)):
                missing_triplet += 1
                continue

            # 统计各模态帧数
            n_rgb = count_rgb_frames(rgb_dir)
            n_dep = count_depth_frames(dep_dir)
            flow_mode, n_flo = count_flow_frames(flo_dir)
            n = min(n_rgb, n_dep, n_flo)

            if n < MIN_FRAMES:
                too_short += 1
                continue

            rel_id = f"{cls}/{vid}"
            label0 = class2label[cls]
            samples.append((rel_id, n, label0, subject))

        if ci % 10 == 0 or ci == len(common_classes):
            print(f"  ...processed classes {ci}/{len(common_classes)}")

    return samples, bad_name, too_short, missing_triplet


def write_txt(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for rel_id, n, label0 in items:
            f.write(f"{rel_id} {n} {label0}\n")


def main():
    print("=" * 80)
    print("THU-READ 三模态 Cross-Subject 列表生成（A方案：适配你的实际数据）")
    print("=" * 80)
    print("[ROOT ]", ROOT)
    print("[RGB  ]", RGB_ROOT)
    print("[DEP  ]", DEP_ROOT)
    print("[FLOW ]", FLOW_ROOT)
    print("[OUT  ]", OUT_DIR)
    print(f"[CFG  ] MIN_FRAMES={MIN_FRAMES} | SEED={SEED}")
    print("-" * 80)

    # 检查核心文件夹是否存在
    for p in [RGB_ROOT, DEP_ROOT, FLOW_ROOT]:
        if not osp.isdir(p):
            raise FileNotFoundError(f"Missing dir: {p}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # 构建共同类别
    common_classes = build_common_classes()
    print(f"[CLASS] detected classes = {len(common_classes)} (当前数据共{len(common_classes)}类)")
    if len(common_classes) == 0:
        print("[ERR] 没有找到三者共同的类别文件夹（请检查 data/RGB, data/Depth, data/FlowRGB）")
        return

    # 构建样本列表
    samples_raw, bad_name, too_short, missing_triplet = build_samples(common_classes)
    print("-" * 80)
    print(f"[SCAN] total usable samples = {len(samples_raw)}")

    # 统计 subject 集合
    subjects = sorted(set(s[3] for s in samples_raw))
    print(f"[SCAN] detected subjects    = {subjects} (count={len(subjects)})")

    # 固定划分检查
    all_split = set(TRAIN_SUBJECTS + VAL_SUBJECTS + TEST_SUBJECTS)
    missing_sub = [s for s in all_split if s not in subjects]
    if missing_sub:
        print("[WARN] 设定的划分里有subject未在数据中找到：", missing_sub)
        print("       这部分subject对应的split会为空。")

    print(f"[SPLIT] train subjects = {TRAIN_SUBJECTS}")
    print(f"[SPLIT] val   subject  = {VAL_SUBJECTS}")
    print(f"[SPLIT] test  subject  = {TEST_SUBJECTS}")
    print("-" * 80)

    # 按 subject 分配样本
    train = []
    val = []
    test = []
    dropped_split = 0

    for rel_id, n, label0, subject in samples_raw:
        item = (rel_id, n, label0)
        if subject in TRAIN_SUBJECTS:
            train.append(item)
        elif subject in VAL_SUBJECTS:
            val.append(item)
        elif subject in TEST_SUBJECTS:
            test.append(item)
        else:
            dropped_split += 1

    # 打印统计信息
    print(
        f"[DROP] bad_name={bad_name} | too_short={too_short} | missing_triplet={missing_triplet} | not_in_split={dropped_split}")
    print(f"[OUT ] train={len(train)} | val={len(val)} | test={len(test)}")

    # 写入文件
    train_file = osp.join(OUT_DIR, "train_universal.txt")
    val_file = osp.join(OUT_DIR, "val_universal.txt")
    test_file = osp.join(OUT_DIR, "test_universal.txt")

    write_txt(train_file, train)
    write_txt(val_file, val)
    write_txt(test_file, test)

    print("[OK] 已生成列表文件：")
    print("  ", train_file)
    print("  ", val_file)
    print("  ", test_file)

    # 打印前5行训练数据（验证格式）
    if train:
        print("\n[CHECK] first 5 train lines:")
        for line in train[:5]:
            print(" ", f"{line[0]} {line[1]} {line[2]}")
    elif val:
        print("\n[CHECK] first 5 val lines:")
        for line in val[:5]:
            print(" ", f"{line[0]} {line[1]} {line[2]}")
    else:
        print("\n[INFO] 暂无训练/验证样本（请检查Subject划分是否正确）")


if __name__ == "__main__":
    main()