# extract_thuread_flow_color_from_rgb_videos.py
import os
import cv2
import sys
import numpy as np
import time
import glob
from pathlib import Path

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".m4v"}

# 你的路径（按你要求写死）
VIDEO_ROOT = r"E:\transformer实验代码\DSCMT-main\data\RGB1"
FLOW_ROOT  = r"E:\transformer实验代码\DSCMT-main\data\FlowRGB"

# 断点续跑进度文件
PROGRESS_FILE = r"E:\transformer实验代码\DSCMT-main\extract_progress_thuread_flow_color.txt"


def imwrite_unicode(path, img):
    """更稳的保存（兼容中文路径）"""
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def keep_largest_cc(mask):
    """只保留最大连通域（通常是主要运动区域）"""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_id = np.argmax(areas) + 1
    return ((labels == keep_id).astype(np.uint8) * 255)


def flow_to_rgb_clean(flow,
                      gate_percentile=95,
                      min_thr=0.20,
                      morph_ksize=3,
                      morph_iter=1):
    """
    把光流转成“彩色图”（HSV 可视化），并做简单去噪
    Hue: 方向（angle）  Saturation: 固定 255  Value: 速度大小（magnitude）
    """
    u, v = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)

    # 运动门控：只显示较大的运动
    thr_p = np.percentile(mag, gate_percentile)
    thr = max(thr_p, min_thr)
    mask = (mag >= thr).astype(np.uint8) * 255

    # 形态学开运算去噪
    kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iter)

    # 只保留最大运动连通域（可注释掉此行，显示所有运动）
    mask = keep_largest_cc(mask)

    # Value 映射（亮度）
    mag2 = mag * (mask > 0)
    p95 = np.percentile(mag2[mag2 > 0], 95) if np.any(mag2 > 0) else 1.0
    val = np.zeros_like(mag2, dtype=np.uint8)
    if p95 < 1e-6:
        p95 = 1.0
    val[mask > 0] = np.clip(mag2[mask > 0] / p95 * 255, 0, 255)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)  # OpenCV Hue: 0~179
    hsv[..., 1] = 255
    hsv[..., 2] = val

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def list_videos_recursive(root_dir):
    vids = []
    for root, _, files in os.walk(root_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                vids.append(os.path.join(root, fn))
    return sorted(vids)


def process_video(video_path, out_dir,
                  resize_wh=(960, 540),
                  every_n=1,
                  max_frames=0,
                  gate_percentile=95,
                  min_thr=0.20):
    """
    输出彩色 flow 帧：flow_00001.png, flow_00002.png, ...
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return 0

    if resize_wh:
        prev = cv2.resize(prev, resize_wh)

    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_g = cv2.GaussianBlur(prev_g, (5, 5), 0)

    idx = 0
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 隔帧处理
        if every_n > 1 and (frame_idx % every_n != 0):
            continue

        if resize_wh:
            frame = cv2.resize(frame, resize_wh)

        cur_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur_g = cv2.GaussianBlur(cur_g, (5, 5), 0)

        flow = cv2.calcOpticalFlowFarneback(
            prev_g, cur_g, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        idx += 1
        flow_rgb = flow_to_rgb_clean(
            flow,
            gate_percentile=gate_percentile,
            min_thr=min_thr,
            morph_ksize=3,
            morph_iter=1
        )

        out_path = os.path.join(out_dir, f"flow_{idx:05d}.png")
        ok = imwrite_unicode(out_path, flow_rgb)
        if not ok:
            # fallback
            ok2 = cv2.imwrite(out_path, flow_rgb)
            if not ok2:
                print(f"[WARN] 保存失败: {out_path}")

        prev_g = cur_g

        if idx % 50 == 0:
            print(f"    ...已生成 {idx} 帧(flow彩色)")

        if max_frames > 0 and idx >= max_frames:
            break

    cap.release()
    return idx


def main():
    print("=" * 80)
    print("THU-READ RGB 视频 -> 彩色光流帧 FlowRGB 批量生成")
    print("=" * 80)
    print(f"[IN ] {VIDEO_ROOT}")
    print(f"[OUT] {FLOW_ROOT}")
    print("-" * 80)

    if not os.path.exists(VIDEO_ROOT):
        print(f"[ERR] 输入目录不存在: {VIDEO_ROOT}")
        return

    os.makedirs(FLOW_ROOT, exist_ok=True)

    videos = list_videos_recursive(VIDEO_ROOT)
    print(f"[SCAN] 找到视频数: {len(videos)}")
    if len(videos) == 0:
        print("[ERR] 没有找到视频文件")
        return

    processed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            processed = set([line.strip() for line in f if line.strip()])
        print(f"[RESUME] 已处理: {len(processed)}")

    # --------- 可调参数 ----------
    RESIZE_WH = (960, 540)   # 提速可改 (640,360) 或 (320,240)
    EVERY_N   = 1            # 隔帧处理：2=隔一帧算一次
    MAX_FRAMES = 0           # 每个视频最多生成多少帧（0不限制）
    GATE_P = 85              # 背景越黑可调 96~97
    MIN_THR = 0.15           # 人动越亮可调 0.15
    # ---------------------------

    t0 = time.time()
    ok_cnt, skip_cnt, fail_cnt, total = 0, 0, 0, 0

    for i, vp in enumerate(videos, 1):
        rel = os.path.relpath(vp, VIDEO_ROOT).replace("\\", "/")
        action_class = rel.split("/")[0]
        stem = Path(vp).stem

        out_dir = os.path.join(FLOW_ROOT, action_class, stem)

        # 已处理且输出足够多帧则跳过
        if rel in processed and os.path.exists(out_dir):
            flows = glob.glob(os.path.join(out_dir, "flow_*.png"))
            if len(flows) > 10:
                skip_cnt += 1
                continue

        print(f"\n[{i}/{len(videos)}] {rel}")
        print(f"  -> {out_dir}")

        n = process_video(
            vp, out_dir,
            resize_wh=RESIZE_WH,
            every_n=EVERY_N,
            max_frames=MAX_FRAMES,
            gate_percentile=GATE_P,
            min_thr=MIN_THR
        )

        if n > 0:
            ok_cnt += 1
            total += n
            with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
                f.write(rel + "\n")
            processed.add(rel)
            print(f"  [OK] flow帧数={n}")
        else:
            fail_cnt += 1
            print(f"  [FAIL] {rel}")

        if i % 20 == 0:
            elapsed = time.time() - t0
            print(f"[STAT] ok={ok_cnt} skip={skip_cnt} fail={fail_cnt} total_flow_frames={total} | {elapsed/60:.1f} min")

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("[DONE] 彩色光流生成完成")
    print(f"ok={ok_cnt} | skip={skip_cnt} | fail={fail_cnt}")
    print(f"total_flow_frames={total}")
    print(f"elapsed={elapsed/60:.2f} minutes")
    print(f"flow_root={FLOW_ROOT}")
    print(f"[NOTE] 进度文件保留：{PROGRESS_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        import cv2
        print(f"[OK] OpenCV version: {cv2.__version__}")
    except ImportError:
        print("[ERR] 请安装 opencv-python：pip install opencv-python")
        sys.exit(1)

    main()
