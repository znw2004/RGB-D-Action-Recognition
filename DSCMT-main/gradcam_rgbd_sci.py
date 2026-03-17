import os
import glob
import argparse
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from DSCMT import TSN


# -------------------------
# IO (unicode safe)
# -------------------------
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
    if img is not None:
        return img
    try:
        pil = Image.open(path)
        if flags == cv2.IMREAD_COLOR:
            pil = pil.convert("RGB")
            arr = np.array(pil)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr
        return np.array(pil)
    except Exception:
        return None


def pick_first_file(path_or_dir, patterns):
    if os.path.isfile(path_or_dir):
        return path_or_dir
    if not os.path.isdir(path_or_dir):
        raise FileNotFoundError(f"path not exists (file/dir): {path_or_dir}")
    for p in patterns:
        files = sorted(glob.glob(os.path.join(path_or_dir, p)))
        if files:
            return files[0]
    raise FileNotFoundError(f"no file matched in dir: {path_or_dir} patterns={patterns}")


# -------------------------
# transforms (match TSN style)
# -------------------------
def resize_shorter_side(img, shorter):
    h, w = img.shape[:2]
    if h < w:
        new_h = shorter
        new_w = int(round(w * (shorter / h)))
    else:
        new_w = shorter
        new_h = int(round(h * (shorter / w)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def center_crop(img, size):
    h, w = img.shape[:2]
    th, tw = size, size
    i = max(0, int(round((h - th) / 2.0)))
    j = max(0, int(round((w - tw) / 2.0)))
    return img[i:i + th, j:j + tw]


def normalize_mean_std(mean, std, n_channels):
    mean = list(mean)
    std = list(std)
    if len(mean) == n_channels and len(std) == n_channels:
        return mean, std
    if len(mean) == 3 and n_channels == 6:
        return mean * 2, std * 2
    if len(mean) == 1 and len(std) == 1:
        return mean * n_channels, std * n_channels
    raise ValueError(f"Unsupported mean/std length: mean={len(mean)} std={len(std)} for n_channels={n_channels}")


def preprocess_rgb_depth(rgb_path_or_dir, depth_path_or_dir, net_input_size, net_scale_size, input_mean, input_std, device):
    rgb_path = pick_first_file(rgb_path_or_dir, ["img_*.jpg", "img_*.png", "*.jpg", "*.png"])
    depth_path = pick_first_file(depth_path_or_dir, ["MDepth-*.png", "MDepth-*.jpg", "*.png", "*.jpg"])

    print("[INFO] picked RGB:", rgb_path)
    print("[INFO] picked Depth:", depth_path)

    rgb_bgr = imread_unicode(rgb_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"RGB not readable: {rgb_path}")

    dep = imread_unicode(depth_path, cv2.IMREAD_UNCHANGED)
    if dep is None:
        raise FileNotFoundError(f"Depth not readable: {depth_path}")

    # depth -> 3ch uint8
    if dep.ndim == 2:
        dep8 = (dep / 256).astype(np.uint8) if dep.dtype == np.uint16 else dep.astype(np.uint8)
        dep3 = np.stack([dep8, dep8, dep8], axis=2)
    else:
        dep3 = dep[:, :, :3]
        dep3 = (dep3 / 256).astype(np.uint8) if dep3.dtype == np.uint16 else dep3.astype(np.uint8)

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    # resize+crop
    rgb_r = center_crop(resize_shorter_side(rgb, net_scale_size), net_input_size)
    dep_r = center_crop(resize_shorter_side(dep3, net_scale_size), net_input_size)

    # 6ch
    x = np.concatenate([rgb_r, dep_r], axis=2).astype(np.float32) / 255.0

    mean6, std6 = normalize_mean_std(input_mean, input_std, 6)
    mean = np.array(mean6, dtype=np.float32).reshape(1, 1, 6)
    std = np.array(std6, dtype=np.float32).reshape(1, 1, 6)
    x = (x - mean) / std

    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    return rgb_r, dep_r, x


# -------------------------
# Grad-CAM core
# -------------------------
class GradCAM:
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        self.h1 = target_module.register_forward_hook(self._forward_hook)
        self.h2 = target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, m, inp, out):
        self.activations = out

    def _backward_hook(self, m, gin, gout):
        # gout: tuple, take first
        self.gradients = gout[0]

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def cam(self, score):
        """
        score: scalar (logit of target class)
        returns: cam map in [0,1], shape HxW of activation
        """
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        A = self.activations  # [B,C,h,w]
        G = self.gradients    # [B,C,h,w]
        if A is None or G is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients. Check target layer.")

        w = G.mean(dim=(2, 3), keepdim=True)           # [B,C,1,1]
        cam = (w * A).sum(dim=1, keepdim=False)        # [B,h,w]
        cam = F.relu(cam)
        cam = cam[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


def overlay(rgb_uint8, heat01, alpha=0.45, cmap=cv2.COLORMAP_JET):
    heat = cv2.resize(heat01, (rgb_uint8.shape[1], rgb_uint8.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat_u8 = np.uint8(255 * np.clip(heat, 0, 1))
    heat_color = cv2.applyColorMap(heat_u8, cmap)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb_uint8, 1 - alpha, heat_color, alpha, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--rgb", required=True, help="RGB file OR dir")
    ap.add_argument("--depth", required=True, help="Depth file OR dir")
    ap.add_argument("--out_prefix", required=True)

    ap.add_argument("--num_classes", type=int, default=40)
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--crop_fusion_type", default="avg")

    ap.add_argument("--target_class", type=int, default=-1, help="-1 means use predicted class")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = TSN(args.num_classes, num_segments=1, modality="Appearance",
              base_model=args.arch, consensus_type=args.crop_fusion_type, dropout=args.dropout)
    net.to(device)
    net.eval()

    ckpt = torch.load(args.weights, map_location=device)
    sd = strip_module_prefix(ckpt["state_dict"])
    net.load_state_dict(sd, strict=True)
    print("[OK] weights loaded:", os.path.basename(args.weights))

    input_mean = getattr(net, "input_mean", [0.485, 0.456, 0.406])
    input_std = getattr(net, "input_std", [0.229, 0.224, 0.225])
    net_input_size = int(getattr(net, "input_size", getattr(net, "crop_size", 224)))
    net_scale_size = int(getattr(net, "scale_size", int(net_input_size * 256 / 224)))

    rgb_vis, dep_vis, x = preprocess_rgb_depth(
        args.rgb, args.depth, net_input_size, net_scale_size, input_mean, input_std, device
    )

    # 关键：对两个 stream 分别做 Grad-CAM
    # ResNet50 的最后卷积块一般是 layer4；你这个 repo 的 base_models 是两个 backbone
    rgb_layer = net.base_models[0].layer4
    dep_layer = net.base_models[1].layer4

    cam_rgb = GradCAM(net, rgb_layer)
    cam_dep = GradCAM(net, dep_layer)

    # forward: net(x) 会返回 (rgb_logit, dep_logit, fuse_logit) 或类似
    out = net(x)
    if isinstance(out, (list, tuple)) and len(out) >= 3:
        rgb_logit, dep_logit, fuse_logit = out[0], out[1], out[2]
    else:
        raise RuntimeError("Unexpected model output. Expected (rgb, depth, fuse) logits tuple.")

    if args.target_class < 0:
        cls = int(torch.argmax(fuse_logit, dim=1).item())
    else:
        cls = int(args.target_class)

    print("[INFO] target class =", cls)

    # 用 fusion logit 作为目标，更符合你论文里“融合后决策”的解释
    score = fuse_logit[0, cls]

    # 为了分别得到两个 stream 的 CAM：对同一个 score 做 backward，
    # cam() 内部会 backward，所以需要 retain_graph
    heat_rgb = cam_rgb.cam(score)
    # 第二次 backward 需要重新算一次 score（或者 retain_graph=True 已经在 cam() 用了）
    score2 = fuse_logit[0, cls]
    heat_dep = cam_dep.cam(score2)

    cam_rgb.close()
    cam_dep.close()

    ov_rgb = overlay(rgb_vis, heat_rgb, alpha=args.alpha, cmap=cv2.COLORMAP_JET)
    ov_dep = overlay(dep_vis, heat_dep, alpha=args.alpha, cmap=cv2.COLORMAP_JET)

    # 论文风格：2x2
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(rgb_vis)
    ax1.set_title("RGB Frame")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(dep_vis, cmap="gray")
    ax2.set_title("Depth Frame")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(ov_rgb)
    ax3.set_title("RGB Grad-CAM (fuse)")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(ov_dep)
    ax4.set_title("Depth Grad-CAM (fuse)")
    ax4.axis("off")

    fig.tight_layout()
    fig.savefig(args.out_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(args.out_prefix + ".png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("[DONE] saved:")
    print("   ", args.out_prefix + ".pdf")
    print("   ", args.out_prefix + ".png")


if __name__ == "__main__":
    main()
