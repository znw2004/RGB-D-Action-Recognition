# extract_thuread_depth_videos_to_frames.py
import os
import sys
import time
import glob
import cv2
from pathlib import Path

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".m4v"}


def to_uint8_gray(frame):
    """
    把各种可能的 depth frame（uint8/uint16/float/3通道）统一转成 uint8 灰度图
    - 如果是 3 通道：转灰度
    - 如果是 uint16/float：min-max 归一化到 0~255
    """
    if frame is None:
        return None

    # BGR/RGB -> Gray
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 已经是 uint8 直接返回
    if frame.dtype == "uint8":
        return frame

    # uint16 / float / 其他：归一化到 0~255
    f = frame.astype("float32")
    mn = float(f.min())
    mx = float(f.max())

    if mx <= mn + 1e-6:
        return (f * 0).astype("uint8")

    f = (f - mn) / (mx - mn) * 255.0
    f = f.clip(0, 255)
    return f.astype("uint8")


def extract_depth_frames(video_path: str, output_dir: str, every_n=1, max_frames=0):
    """
    深度视频 -> 帧序列（鲁棒版）
    输出命名：MDepth-00000001.png（与 dataset.py 兼容）
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERR] 无法打开视频: {video_path}")
        return 0

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 隔帧采样
        if every_n > 1 and (frame_idx % every_n != 0):
            continue

        saved += 1
        out_name = f"MDepth-{saved:08d}.png"
        out_path = os.path.join(output_dir, out_name)

        img8 = to_uint8_gray(frame)
        if img8 is None:
            print(f"[WARN] 空帧: {out_path}")
            continue

        ok = cv2.imwrite(out_path, img8)
        if not ok:
            # fallback: PIL 再试一次
            try:
                from PIL import Image
                Image.fromarray(img8).save(out_path)
                ok = True
            except Exception as e:
                # 打印调试信息
                try:
                    dtype = str(frame.dtype) if frame is not None else "None"
                    shape = str(frame.shape) if frame is not None else "None"
                    fmin = float(frame.min()) if frame is not None else None
                    fmax = float(frame.max()) if frame is not None else None
                except Exception:
                    dtype, shape, fmin, fmax = "?", "?", "?", "?"
                print(f"[WARN] 保存失败: {out_path} | dtype={dtype} shape={shape} min={fmin} max={fmax} | {e}")

        if saved % 200 == 0:
            print(f"    ...已保存 {saved} 帧")

        if max_frames > 0 and saved >= max_frames:
            break

    cap.release()
    return saved


def list_videos_recursive(root_dir: str):
    videos = []
    for root, _, files in os.walk(root_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                videos.append(os.path.join(root, fn))
    return sorted(videos)


def main():
    print("=" * 80)
    print("THU-READ Depth 视频 -> Depth 帧序列（MDepth-xxxxxxxx.png）")
    print("=" * 80)

    start_time = time.time()

    # 固定为你的工程根目录（与你实际路径一致）
    project_root = r"E:\transformer实验代码\DSCMT-main"

    # 输入：Depth1（深度视频）
    depth_video_root = os.path.join(project_root, "data", "Depth1")

    # 输出：Depth（深度帧）
    out_depth_root = os.path.join(project_root, "data", "Depth")

    # 断点续跑记录
    progress_file = os.path.join(project_root, "extract_progress_thuread_depth.txt")

    # 可调：隔帧采样（1=全存；2=隔帧）
    EVERY_N = 1

    # 可调：每个视频最多保存多少帧（0=不限制）
    MAX_FRAMES = 0

    print(f"[ROOT] project_root     = {project_root}")
    print(f"[IN  ] Depth1 videos    = {depth_video_root}")
    print(f"[OUT ] Depth frames     = {out_depth_root}")
    print(f"[CFG ] every_n={EVERY_N} | max_frames={MAX_FRAMES}")
    print("-" * 80)

    if not os.path.exists(depth_video_root):
        print(f"[ERR] 输入目录不存在: {depth_video_root}")
        return

    os.makedirs(out_depth_root, exist_ok=True)

    videos = list_videos_recursive(depth_video_root)
    print(f"[SCAN] 找到深度视频数: {len(videos)}")
    if len(videos) == 0:
        print("[ERR] 没有找到深度视频文件（检查后缀或目录）")
        return

    processed = set()
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            processed = set([line.strip() for line in f if line.strip()])
        print(f"[RESUME] 已处理: {len(processed)}")

    ok_cnt, skip_cnt, fail_cnt, total_saved = 0, 0, 0, 0

    for i, vp in enumerate(videos, 1):
        # 相对路径：bounce_ball\D_bounce_ball_cgy_1.avi
        rel = os.path.relpath(vp, depth_video_root).replace("\\", "/")
        action_class = rel.split("/")[0]
        stem = Path(vp).stem  # D_bounce_ball_cgy_1

        # 输出目录：Depth/<class>/<stem>/
        out_dir = os.path.join(out_depth_root, action_class, stem)

        # 断点：存在输出且帧数足够就跳过
        if rel in processed and os.path.exists(out_dir):
            pngs = glob.glob(os.path.join(out_dir, "MDepth-*.png"))
            if len(pngs) > 10:
                skip_cnt += 1
                continue

        print(f"\n[{i}/{len(videos)}] {rel}")
        saved = extract_depth_frames(vp, out_dir, every_n=EVERY_N, max_frames=MAX_FRAMES)

        if saved > 0:
            ok_cnt += 1
            total_saved += saved
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(rel + "\n")
            processed.add(rel)
        else:
            fail_cnt += 1
            print(f"[WARN] 提取失败: {rel}")

        if i % 20 == 0:
            elapsed = time.time() - start_time
            print(f"[STAT] ok={ok_cnt} skip={skip_cnt} fail={fail_cnt} saved_frames={total_saved}")
            print(f"[TIME] elapsed={elapsed/60:.1f} min")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("[DONE] 深度帧提取完成")
    print(f"ok={ok_cnt} | skip={skip_cnt} | fail={fail_cnt}")
    print(f"total_saved_frames={total_saved}")
    print(f"elapsed={elapsed/60:.2f} minutes")
    print(f"depth_frames_root={out_depth_root}")
    print("=" * 80)
    print(f"[NOTE] 进度文件保留：{progress_file}")


if __name__ == "__main__":
    try:
        import cv2
        print(f"[OK] OpenCV version: {cv2.__version__}")
    except ImportError:
        print("[ERR] 请先安装 OpenCV：pip install opencv-python")
        sys.exit(1)

    main()
