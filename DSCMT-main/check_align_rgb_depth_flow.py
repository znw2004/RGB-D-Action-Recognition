import os
import glob
import random
import cv2
import numpy as np

# ====== 你的数据根目录 ======
ROOT = r"E:\transformer实验代码\DSCMT-main\data"
RGB_ROOT = os.path.join(ROOT, "RGB")
DEPTH_ROOT = os.path.join(ROOT, "Depth")
FLOW_ROOT = os.path.join(ROOT, "FlowRGB")

# ====== 帧文件模式（按你当前生成的格式） ======
RGB_GLOB = "img_*.jpg"
DEPTH_GLOB = "MDepth-*.png"
FLOW_GLOB = "flow_*.png"  # 你现在生成的是彩色 flow_00001.png

SHOW_W = 480  # 每个模态显示宽度


def imread_unicode(path, flag=cv2.IMREAD_UNCHANGED):
    """支持中文路径读取"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flag)


def resize_keep_aspect(img, target_w):
    h, w = img.shape[:2]
    if w <= 0:
        return img
    scale = target_w / float(w)
    nh = max(1, int(h * scale))
    return cv2.resize(img, (target_w, nh))


def depth_to_vis(depth_img):
    """把深度图转成更好看的伪彩色，支持 8bit/16bit/3通道。"""
    if depth_img is None:
        return None

    if len(depth_img.shape) == 3:
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

    if depth_img.dtype == np.uint16:
        d = depth_img.astype(np.float32)
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        if hi <= lo:
            hi = lo + 1.0
        d = np.clip((d - lo) / (hi - lo), 0, 1) * 255.0
        d8 = d.astype(np.uint8)
    else:
        d8 = depth_img.astype(np.uint8)

    return cv2.applyColorMap(d8, cv2.COLORMAP_JET)


def normalize_video_name(name: str) -> str:
    """
    关键：统一 THU-READ 三模态视频文件夹命名
      RGB/Flow: RGB_bounce_ball_cgy_1  -> bounce_ball_cgy_1
      Depth:    D_bounce_ball_cgy_1    -> bounce_ball_cgy_1
    """
    n = name
    for p in ["RGB_", "D_", "F_", "DEPTH_", "FLOW_"]:
        if n.startswith(p):
            n = n[len(p):]
            break
    return n


def scan_leaf_video_dirs(mod_root: str, frame_glob: str):
    """
    递归扫描：找到包含帧文件的“叶子文件夹”，并建立 key -> folder_path 映射
    key 使用 (class_name, normalized_video_name)
    class_name 使用第一层目录名（如 bounce_ball）
    """
    mapping = {}
    for dirpath, _, _ in os.walk(mod_root):
        files = glob.glob(os.path.join(dirpath, frame_glob))
        if len(files) == 0:
            continue

        rel = os.path.relpath(dirpath, mod_root)
        parts = rel.split(os.sep)

        # 期望结构：mod_root / class / video
        if len(parts) >= 2:
            cls = parts[0]
            vid = parts[-1]
        else:
            cls = "__root__"
            vid = parts[-1]

        vid_norm = normalize_video_name(vid)
        key = (cls, vid_norm)
        mapping[key] = dirpath

    return mapping


def get_sorted_frames(folder, pattern):
    files = glob.glob(os.path.join(folder, pattern))
    files.sort()
    return files


def main():
    print("=" * 80)
    print("三模态对齐可视化检查（B方案：容错匹配：去掉 RGB_/D_ 前缀）")
    print("=" * 80)
    print("[RGB ]", RGB_ROOT)
    print("[DEP ]", DEPTH_ROOT)
    print("[FLOW]", FLOW_ROOT)
    print("-" * 80)

    for p in [RGB_ROOT, DEPTH_ROOT, FLOW_ROOT]:
        if not os.path.exists(p):
            print(f"[ERR] 目录不存在: {p}")
            return

    rgb_map = scan_leaf_video_dirs(RGB_ROOT, RGB_GLOB)
    dep_map = scan_leaf_video_dirs(DEPTH_ROOT, DEPTH_GLOB)
    flo_map = scan_leaf_video_dirs(FLOW_ROOT, FLOW_GLOB)

    common_keys = sorted(list(set(rgb_map.keys()) & set(dep_map.keys()) & set(flo_map.keys())))

    print(f"[SCAN] rgb_leaf={len(rgb_map)} | depth_leaf={len(dep_map)} | flow_leaf={len(flo_map)}")
    print(f"[SCAN] common samples = {len(common_keys)}")

    if len(common_keys) == 0:
        print("[ERR] 仍然匹配不到三者交集（理论上你现在的结构应该能匹配）。")
        print("你可以对照这三点：")
        print("  1) 三个根目录下的类别文件夹名是否一致（如 bounce_ball）")
        print("  2) RGB/Flow 视频文件夹是否以 RGB_ 开头，Depth 是否以 D_ 开头")
        print("  3) 三个模态的叶子目录里是否确实有对应帧文件（img_ / MDepth- / flow_）")
        return

    # 随机挑一个样本
    key = random.choice(common_keys)
    cls, vid_norm = key

    def load_sample(k):
        c, v = k
        rgb_dir = rgb_map[k]
        dep_dir = dep_map[k]
        flo_dir = flo_map[k]
        rgb_list = get_sorted_frames(rgb_dir, RGB_GLOB)
        dep_list = get_sorted_frames(dep_dir, DEPTH_GLOB)
        flo_list = get_sorted_frames(flo_dir, FLOW_GLOB)
        n = min(len(rgb_list), len(dep_list), len(flo_list))
        return c, v, rgb_dir, dep_dir, flo_dir, rgb_list, dep_list, flo_list, n

    cls, vid_norm, rgb_dir, dep_dir, flo_dir, rgb_list, dep_list, flo_list, n = load_sample(key)

    print(f"[PICK] class={cls} | video(norm)={vid_norm}")
    print(f"[PATH] RGB  ={rgb_dir}")
    print(f"[PATH] DEPTH={dep_dir}")
    print(f"[PATH] FLOW ={flo_dir}")
    print(f"[CNT ] RGB={len(rgb_list)} DEPTH={len(dep_list)} FLOW={len(flo_list)} -> use min={n}")
    print("操作：A/←上一帧，D/→下一帧，Q退出，R随机换样本")

    if n == 0:
        print("[ERR] 这个样本 min_frames=0（某个模态没帧）")
        return

    idx = 0
    while True:
        rgb = imread_unicode(rgb_list[idx], cv2.IMREAD_COLOR)
        dep_raw = imread_unicode(dep_list[idx], cv2.IMREAD_UNCHANGED)
        flo = imread_unicode(flo_list[idx], cv2.IMREAD_COLOR)

        dep = depth_to_vis(dep_raw)

        if rgb is None or dep is None or flo is None:
            print("[WARN] 读图失败，自动跳下一帧")
            idx = (idx + 1) % n
            continue

        rgb_s = resize_keep_aspect(rgb, SHOW_W)
        dep_s = resize_keep_aspect(dep, SHOW_W)
        flo_s = resize_keep_aspect(flo, SHOW_W)

        h = max(rgb_s.shape[0], dep_s.shape[0], flo_s.shape[0])

        def pad_to_h(img):
            if img.shape[0] == h:
                return img
            pad = h - img.shape[0]
            return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        rgb_s = pad_to_h(rgb_s)
        dep_s = pad_to_h(dep_s)
        flo_s = pad_to_h(flo_s)

        show = np.hstack([rgb_s, dep_s, flo_s])

        title = f"{cls}/{vid_norm} | idx={idx+1:04d}/{n} | RGB | DEPTH | FLOW"
        cv2.putText(show, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("ALIGN CHECK: RGB | DEPTH | FLOW", show)
        keycode = cv2.waitKey(0) & 0xFF

        if keycode in (ord('q'), 27):
            break
        elif keycode in (ord('a'), 81):  # A 或 ←
            idx = (idx - 1) % n
        elif keycode in (ord('d'), 83):  # D 或 →
            idx = (idx + 1) % n
        elif keycode == ord('r'):
            key = random.choice(common_keys)
            cls, vid_norm, rgb_dir, dep_dir, flo_dir, rgb_list, dep_list, flo_list, n = load_sample(key)
            idx = 0
            print(f"\n[NEW] class={cls} | video(norm)={vid_norm} | min_frames={n}")
            print(f"[PATH] RGB  ={rgb_dir}")
            print(f"[PATH] DEPTH={dep_dir}")
            print(f"[PATH] FLOW ={flo_dir}")
        else:
            idx = (idx + 1) % n

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
