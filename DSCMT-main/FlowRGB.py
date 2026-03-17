import os
import cv2
import numpy as np
import time

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")

# 你的路径
video_root = r"E:\transformer实验代码\DSCMT-main\data\RGB1"
flow_root  = r"E:\transformer实验代码\DSCMT-main\data\FlowRGB"


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def keep_largest_cc(mask):
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
    u, v = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)

    thr_p = np.percentile(mag, gate_percentile)
    thr = max(thr_p, min_thr)
    mask = (mag >= thr).astype(np.uint8) * 255

    # 去噪
    kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iter)

    # 只保留最大运动连通域（人）
    mask = keep_largest_cc(mask)

    # 亮度映射
    mag2 = mag * (mask > 0)
    p95 = np.percentile(mag2[mag2 > 0], 95) if np.any(mag2 > 0) else 1.0
    val = np.zeros_like(mag2, dtype=np.uint8)
    val[mask > 0] = np.clip(mag2[mask > 0] / p95 * 255, 0, 255)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = val

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def process_video(video_path, out_dir,
                  resize_wh=None,
                  gate_percentile=95,
                  min_thr=0.20):

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    ret, prev = cap.read()
    if not ret:
        return

    if resize_wh:
        prev = cv2.resize(prev, resize_wh)

    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_g = cv2.GaussianBlur(prev_g, (5, 5), 0)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
        imwrite_unicode(out_path, flow_rgb)

        prev_g = cur_g

        if idx % 50 == 0:
            print(f"    已生成 {idx} 帧")

    cap.release()
    print(f"    完成：{idx} 帧")


def main():
    t0 = time.time()
    videos = [f for f in os.listdir(video_root) if f.lower().endswith(VIDEO_EXTS)]
    videos.sort()

    print("视频数量：", len(videos))

    for i, v in enumerate(videos, 1):
        name = os.path.splitext(v)[0]
        vpath = os.path.join(video_root, v)
        out_dir = os.path.join(flow_root, name)

        print(f"\n[{i}/{len(videos)}] 处理 {v}")
        print("输出到：", out_dir)

        process_video(
            vpath,
            out_dir,
            resize_wh=(960,540),      # 若想提速可改成 (320,240)
            gate_percentile=95, # 背景越黑可调到 96~97
            min_thr=0.20        # 人动越亮可调到 0.15
        )

    print("\n总耗时：", (time.time() - t0) / 60, "分钟")


if __name__ == "__main__":
    main()
