import os
import time
import cv2
import numpy as np

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")

# ========= 你的路径 =========
video_root = r"E:\transformer实验代码\DSCMT-main\data\RGB1"    # RGB视频目录
flow_root  = r"E:\transformer实验代码\DSCMT-main\data\FlowRGB" # 输出目录（每个视频一个子文件夹）
# ===========================

# ========= 输出文件名规则（需与 main.py 的 image_tmpl 匹配）=========
FLOW_PREFIX = "flow_"   # 对应训练时：--flow_prefix flow_
OUT_EXT = ".jpg"        # main.py 默认模板是 .jpg；请保持一致
JPG_QUALITY = 95
# ================================================================

# ========= 速度/质量（推荐你现在用 960x540）=========
RESIZE_WH = (960, 540)     # None=原分辨率（很慢）；建议 (960,540) 或 (640,480)
# ===================================================

# ========= 背景抑制（可选：更“只剩人”）=========
USE_MOTION_MASK = True     # True: 背景近似零流（128）更干净；False: 不做mask，保留全图流
GATE_PERCENTILE = 95       # 95~97 越大背景越干净但越可能吃掉细动作
MIN_MAG_THR = 0.20         # 0.15~0.35 视你数据而定
MORPH_KSIZE = 3            # 3 或 5
MORPH_ITER = 1
# ===================================================

# ========= 分量编码（非常重要）=========
# flow_x/flow_y 存储为 0~255，128 表示 0 流。
# bound 越大 -> 编码越不饱和（更稳），越小 -> 更亮但更易饱和。
BOUND_MIN = 8.0            # 下限，防止静态时 bound 太小导致噪声被放大
BOUND_PCTL = 99            # 用 abs(u/v) 的百分位做自适应上界（推荐 97~99）
# ======================================


def imwrite_unicode(path, img, jpg_quality=95):
    """Windows 中文路径安全写入"""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        ok, buf = cv2.imencode(ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
    else:
        ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def keep_largest_cc(mask_u8):
    """mask_u8: 0/255，保留最大连通域"""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask_u8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_id = int(np.argmax(areas) + 1)
    return (labels == keep_id).astype(np.uint8) * 255


def build_motion_mask(flow, gate_percentile=95, min_mag_thr=0.20, morph_ksize=3, morph_iter=1):
    """根据 magnitude 门控，得到“运动区域mask”（0/255）"""
    u = flow[..., 0]
    v = flow[..., 1]
    mag = np.sqrt(u * u + v * v)

    thr_p = float(np.percentile(mag, gate_percentile))
    thr = max(thr_p, float(min_mag_thr))
    mask = (mag >= thr).astype(np.uint8) * 255

    if morph_ksize and morph_ksize > 1 and morph_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=morph_iter)

    mask = keep_largest_cc(mask)
    return mask


def encode_flow_component(comp, bound):
    """
    comp: float32 (u or v)
    bound: 正值上界
    输出 uint8：128 表示 0 流
    """
    # [-bound, bound] -> [1, 255] around 128, clamp
    scaled = comp / bound * 127.0 + 128.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def process_one_video(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERR] 无法打开视频: {video_path}")
        return 0

    ret, prev = cap.read()
    if not ret:
        cap.release()
        print(f"[ERR] 读取首帧失败: {video_path}")
        return 0

    if RESIZE_WH is not None:
        prev = cv2.resize(prev, RESIZE_WH)

    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_g = cv2.GaussianBlur(prev_g, (5, 5), 0)

    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if RESIZE_WH is not None:
            frame = cv2.resize(frame, RESIZE_WH)

        cur_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur_g = cv2.GaussianBlur(cur_g, (5, 5), 0)

        flow = cv2.calcOpticalFlowFarneback(
            prev_g, cur_g, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        u = flow[..., 0]
        v = flow[..., 1]

        # 可选：只保留“人体运动区域”，其他区域置零流（=128）
        if USE_MOTION_MASK:
            m = build_motion_mask(flow, GATE_PERCENTILE, MIN_MAG_THR, MORPH_KSIZE, MORPH_ITER)
            keep = (m > 0)
        else:
            keep = None

        # 自适应 bound：避免小噪声被放大，同时避免大运动饱和
        if keep is not None and np.any(keep):
            bu = float(np.percentile(np.abs(u[keep]), BOUND_PCTL))
            bv = float(np.percentile(np.abs(v[keep]), BOUND_PCTL))
        else:
            bu = float(np.percentile(np.abs(u), BOUND_PCTL))
            bv = float(np.percentile(np.abs(v), BOUND_PCTL))

        bu = max(bu, BOUND_MIN)
        bv = max(bv, BOUND_MIN)

        fx = encode_flow_component(u, bu)
        fy = encode_flow_component(v, bv)

        # mask 外强制置零流（128）
        if keep is not None:
            fx[~keep] = 128
            fy[~keep] = 128

        idx += 1  # 第一张流对应 frame1->frame2，编号从1开始最常见
        x_path = os.path.join(out_dir, f"{FLOW_PREFIX}x_{idx:05d}{OUT_EXT}")
        y_path = os.path.join(out_dir, f"{FLOW_PREFIX}y_{idx:05d}{OUT_EXT}")

        okx = imwrite_unicode(x_path, fx, JPG_QUALITY)
        oky = imwrite_unicode(y_path, fy, JPG_QUALITY)

        if okx and oky:
            saved += 1

        prev_g = cur_g

        if idx % 200 == 0:
            print(f"    ...已生成 {idx} 组 flow（x/y）")

    cap.release()
    return saved


def main():
    print("=" * 80)
    print("RGB视频 -> Flow(x,y) 灰度帧（兼容 DSCMT）")
    print("=" * 80)
    print("video_root:", video_root)
    print("flow_root :", flow_root)
    print("RESIZE_WH :", RESIZE_WH)
    print("USE_MOTION_MASK:", USE_MOTION_MASK)
    print("-" * 80)

    if not os.path.exists(video_root):
        print("[ERR] 输入目录不存在：", video_root)
        return

    os.makedirs(flow_root, exist_ok=True)

    vids = [f for f in os.listdir(video_root) if f.lower().endswith(VIDEO_EXTS)]
    vids.sort()

    print("找到视频数：", len(vids))
    if not vids:
        print("[ERR] 没有视频文件")
        return

    t0 = time.time()
    ok = 0

    for i, vf in enumerate(vids, 1):
        vpath = os.path.join(video_root, vf)
        name = vf[:-8] if vf.endswith("_rgb.avi") else os.path.splitext(vf)[0]
        out_dir = os.path.join(flow_root, name)

        print(f"\n[{i}/{len(vids)}] {vf}")
        print("    输出目录:", out_dir)

        saved = process_one_video(vpath, out_dir)
        print(f"    [OK] 生成 flow 组数 = {saved}（每组=一张x + 一张y）")

        if saved > 0:
            ok += 1

    print("\n" + "=" * 80)
    print(f"完成：成功 {ok}/{len(vids)}")
    print(f"总耗时：{(time.time() - t0)/60:.2f} 分钟")
    print("=" * 80)
    print("\n训练时请确保：")
    print("1) 使用 --modality Flow")
    print(f"2) 使用 --flow_prefix {FLOW_PREFIX}")
    print("3) dataset.py 会按 x/y 两张灰度图读取（与你现在生成一致）")


if __name__ == "__main__":
    main()
