import os
import glob
import re
import argparse
import math
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image

from DSCMT import TSN


# ----------------------------
# 0) utils
# ----------------------------
def strip_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[".".join(k.split(".")[1:])] = v
        else:
            out[k] = v
    return out


def imread_unicode_cv2(path, flags):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    img = imread_unicode_cv2(path, flags)
    return img


def pick_first_image(path_or_dir, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    if os.path.isdir(path_or_dir):
        files = []
        for e in exts:
            files += glob.glob(os.path.join(path_or_dir, f"*{e}"))
        files = sorted(files)
        if not files:
            raise FileNotFoundError(f"Directory has no images: {path_or_dir}")
        return files[0]
    return path_or_dir


def normalize_mean_std(input_mean, input_std, n_channels=6):
    # repo 里通常是 3 通道 mean/std，这里扩展到 6 通道（RGB+Depth）
    if len(input_mean) == 3:
        mean6 = list(input_mean) + list(input_mean)
    else:
        mean6 = list(input_mean)
    if len(input_std) == 3:
        std6 = list(input_std) + list(input_std)
    else:
        std6 = list(input_std)
    assert len(mean6) == n_channels and len(std6) == n_channels
    return mean6, std6


def factor_grid(n_patch):
    # 尽量找接近平方的 (h,w)
    h = int(math.sqrt(n_patch))
    while h > 1 and (n_patch % h) != 0:
        h -= 1
    w = n_patch // h
    return h, w


# ----------------------------
# 1) load model
# ----------------------------
def load_model(weights_path, device):
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        epoch = checkpoint.get("epoch", "NA")
        best = checkpoint.get("best_prec1", "NA")
        print(f"model epoch {epoch} best prec@1: {best}")
    else:
        state = checkpoint
        print("checkpoint is not dict with state_dict, using raw")
    state = strip_module_prefix(state)

    # 这里保持与你 test_models.py 一致的构建方式：Appearance + resnet50 + num_segments=1
    net = TSN(
        num_class=40,                 # THU-READ 40 类（你可以用参数改）
        num_segments=1,
        modality="Appearance",
        base_model="resnet50",
        consensus_type="avg",
        dropout=0.5
    )

    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[WARN] unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    net.to(device)
    net.eval()
    print("[OK] weights loaded:", os.path.basename(weights_path))
    return net


# ----------------------------
# 2) preprocess single RGB + Depth frame -> 6ch tensor
# ----------------------------
def preprocess_rgb_depth(rgb_path, depth_path, input_size, scale_size, input_mean, input_std, device):
    rgb_path = pick_first_image(rgb_path)
    depth_path = pick_first_image(depth_path)

    print("[INFO] picked RGB:", rgb_path)
    print("[INFO] picked Depth:", depth_path)

    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB path not exists: {rgb_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth path not exists: {depth_path}")

    rgb = imread_unicode(rgb_path, cv2.IMREAD_COLOR)
    dep = imread_unicode(depth_path, cv2.IMREAD_GRAYSCALE)  # depth 当灰度读

    if rgb is None:
        raise FileNotFoundError(f"RGB image not readable by OpenCV (unicode path issue): {rgb_path}")
    if dep is None:
        raise FileNotFoundError(f"Depth image not readable by OpenCV (unicode path issue): {depth_path}")

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # resize（参考 TSN 的 scale_size/input_size）
    # 这里简化：直接 resize 到 input_size
    rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    dep = cv2.resize(dep, (input_size, input_size), interpolation=cv2.INTER_NEAREST)

    dep3 = np.stack([dep, dep, dep], axis=2)  # (H,W,3)

    x = np.concatenate([rgb, dep3], axis=2).astype(np.float32) / 255.0  # (H,W,6)

    mean6, std6 = normalize_mean_std(input_mean, input_std, n_channels=6)
    mean = np.array(mean6, dtype=np.float32).reshape(1, 1, 6)
    std = np.array(std6, dtype=np.float32).reshape(1, 1, 6)
    x = (x - mean) / std

    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,6,H,W)
    return rgb, dep, x


# ----------------------------
# 3) hooks: capture “attention-like” tensors & print where they come from
# ----------------------------
def looks_like_attn(t: torch.Tensor) -> bool:
    if (not torch.is_tensor(t)) or t.ndim != 4:
        return False
    b, h, n1, n2 = t.shape
    if n1 != n2:
        return False
    if n1 < 8 or n1 > 4096:
        return False
    return True


class AttnCatcher:
    def __init__(self):
        self.records = []  # list of (module_name, tensor)

    def hook(self, name):
        def _fn(module, inp, out):
            candidates = []

            def collect(obj):
                if torch.is_tensor(obj):
                    if looks_like_attn(obj):
                        candidates.append(obj)
                elif isinstance(obj, (list, tuple)):
                    for it in obj:
                        collect(it)
                elif isinstance(obj, dict):
                    for _, v in obj.items():
                        collect(v)

            collect(out)

            if candidates:
                t = candidates[-1].detach()
                self.records.append((name, t))
        return _fn


def register_attn_hooks(net, hook_regex=None):
    catcher = AttnCatcher()
    hooks = []

    pattern = re.compile(hook_regex) if hook_regex else None

    for name, m in net.named_modules():
        if pattern is None:
            hooks.append(m.register_forward_hook(catcher.hook(name)))
        else:
            if pattern.search(name):
                hooks.append(m.register_forward_hook(catcher.hook(name)))

    return catcher, hooks


def print_module_names(net, filter_regex=None):
    pat = re.compile(filter_regex) if filter_regex else None
    for name, m in net.named_modules():
        if pat is None or pat.search(name):
            print(name, "->", m.__class__.__name__)


# ----------------------------
# 4) main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--rgb", required=True, help="RGB image path OR directory")
    parser.add_argument("--depth", required=True, help="Depth image path OR directory")
    parser.add_argument("--out_prefix", required=True)

    parser.add_argument("--num_classes", type=int, default=40)

    # 👇 关键：看 hook 到哪一层
    parser.add_argument("--list_modules", action="store_true",
                        help="print all module names (use --module_filter to narrow)")
    parser.add_argument("--module_filter", type=str, default=None,
                        help="regex to filter modules when listing, e.g. 'mem|mim|attn|Attention'")

    parser.add_argument("--hook_regex", type=str, default=None,
                        help="only hook modules whose name matches regex, e.g. 'mem|mim|attn|Attention'")
    parser.add_argument("--print_records", action="store_true",
                        help="after forward, print captured module names + tensor shapes")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = load_model(args.weights, device)

    if args.list_modules:
        print("========== MODULE LIST ==========")
        print_module_names(net, args.module_filter)
        print("========== END ==========")
        return

    # 这些从 TSN 里取（这里给常用默认值；如果你项目里 TSN 有对应属性，你也可以读 net.input_mean 等）
    net_input_size = 224
    net_scale_size = 256
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    rgb_img, dep_img, x = preprocess_rgb_depth(
        args.rgb, args.depth,
        net_input_size, net_scale_size,
        input_mean, input_std,
        device
    )

    # hooks
    catcher, hooks = register_attn_hooks(net, args.hook_regex)

    with torch.no_grad():
        out = net(x)

    for h in hooks:
        h.remove()

    # out 可能是 (rgb_logits, depth_logits, fuse_logits)
    if isinstance(out, (list, tuple)) and len(out) >= 3:
        fuse_logits = out[2]
    else:
        fuse_logits = out

    pred = int(torch.argmax(fuse_logits, dim=1).item())
    print("[INFO] predicted class id:", pred)

    if args.print_records:
        print("========== HOOK RECORDS ==========")
        if not catcher.records:
            print("[WARN] No attention-like tensors captured.")
        for i, (name, t) in enumerate(catcher.records):
            print(f"[{i:03d}] {name}  shape={tuple(t.shape)}  dtype={t.dtype}")
        print("========== END ==========")

    # 如果你之后要做 MEM/MIM 的 2D attention 可视化：
    # 你先用 --print_records 找到真正的 MEM/MIM attention 输出层名（含 mem/mim 关键字）
    # 再用 --hook_regex 只 hook 那个层，下一步再做 2D 可视化。
    #
    # 这里先把输入帧保存一下，方便你对照
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("RGB Frame")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(dep_img, cmap="gray")
    plt.title("Depth Frame")
    plt.axis("off")

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_prefix + "_inputs.png", dpi=300)
    plt.close()
    print("[OK] saved:", args.out_prefix + "_inputs.png")


if __name__ == "__main__":
    main()
