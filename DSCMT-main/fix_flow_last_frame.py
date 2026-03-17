import os
import shutil
import re

flow_root = r"E:\transformer实验代码\DSCMT-main\data\FlowRGB"

pattern = re.compile(r"flow_(\d+)\.png")

def fix_one_video(folder):
    files = [f for f in os.listdir(folder) if f.startswith("flow_") and f.endswith(".png")]
    if not files:
        return

    nums = sorted([int(pattern.match(f).group(1)) for f in files])
    max_id = nums[-1]
    expected_last = max_id + 1

    src = os.path.join(folder, f"flow_{max_id:05d}.png")
    dst = os.path.join(folder, f"flow_{expected_last:05d}.png")

    if not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"[OK] {folder}: 补齐 flow_{expected_last:05d}.png")
    else:
        print(f"[SKIP] {folder}: 已存在 flow_{expected_last:05d}.png")

for root, dirs, files in os.walk(flow_root):
    if any(f.startswith("flow_") for f in files):
        fix_one_video(root)

print("=== 所有 Flow 视频末帧已补齐 ===")
