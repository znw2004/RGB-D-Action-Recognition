import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def build_class_names_from_list(txt_path, num_classes=None, encoding="utf-8"):
    id2name = {}
    with open(txt_path, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            relpath, lab = parts[0], int(parts[2])
            name = relpath.replace("\\", "/").split("/")[0]
            if lab in id2name and id2name[lab] != name:
                raise ValueError(f"Label {lab} name conflict: {id2name[lab]} vs {name}")
            id2name[lab] = name

    if num_classes is None:
        if not id2name:
            raise ValueError("No valid lines found in list file.")
        num_classes = max(id2name.keys()) + 1

    return [id2name.get(i, f"class_{i}") for i in range(num_classes)]


def _clean_labels(labels):
    """把 labels 清洗成 (N,) int64，兼容 object/(N,1)/one-hot 等"""
    if isinstance(labels, np.ndarray) and labels.dtype == object:
        labels = np.array([int(np.array(x).squeeze()) for x in labels], dtype=np.int64)
    else:
        labels = np.array(labels)

    labels = labels.squeeze()

    # one-hot -> id
    if labels.ndim == 2 and labels.shape[1] > 1:
        labels = labels.argmax(axis=1)

    labels = labels.reshape(-1).astype(np.int64)
    return labels


def _scores_to_video_level(scores, N):
    """
    借鉴你 test_models.py 的逻辑，把保存的 scores 聚合为视频级 (N, C)：
    - 你的脚本：np.argmax(np.mean(x, axis=0)) （x 是某个视频的多段/多crop输出）
    - 我们泛化：对除最后一维(C)之外的维度全部 mean
    兼容 scores：
      - list / object array：len = N，每个元素形状如 (T,1,C)
      - ndarray：(N,1,1,C) / (N,T,C) / (N,C) / (N*T,C)
    """
    # 1) 如果是 list 或 object array，先 stack 成 ndarray
    if isinstance(scores, list):
        scores = np.array(scores, dtype=object)

    if isinstance(scores, np.ndarray) and scores.dtype == object:
        # 每个元素是一个 array/tensor
        elems = [np.array(x) for x in scores.tolist()]
        # 尽量 stack；不行就逐个聚合
        try:
            scores = np.stack(elems, axis=0)
        except Exception:
            # 逐个聚合到 (C,)
            vecs = []
            for x in elems:
                x = np.array(x)
                if x.ndim == 0:
                    raise ValueError("score element is scalar, invalid.")
                # 对除最后一维外的所有维度求 mean
                if x.ndim > 1:
                    x = x.reshape(-1, x.shape[-1]).mean(axis=0)
                vecs.append(x)
            scores = np.stack(vecs, axis=0)

    scores = np.array(scores)

    # 2) 若是 (M, C) 且 M != N，则尝试 reshape 成 (N, T, C) 再 mean
    if scores.ndim == 2:
        M, C = scores.shape
        if M == N:
            return scores
        if M % N != 0:
            raise ValueError(f"scores shape {scores.shape} cannot align with labels length {N}")
        T = M // N
        return scores.reshape(N, T, C).mean(axis=1)

    # 3) 若维度 >=3，统一：保留第0维为 N，最后一维为 C，中间全部展平后 mean
    if scores.ndim >= 3:
        if scores.shape[0] != N:
            raise ValueError(f"scores first dim {scores.shape[0]} != labels length {N}, scores shape={scores.shape}")
        C = scores.shape[-1]
        return scores.reshape(N, -1, C).mean(axis=1)

    raise ValueError(f"Unsupported scores ndim: {scores.ndim}, shape={scores.shape}")


def load_npz_video_preds(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "scores" not in data or "labels" not in data:
        raise KeyError("npz must contain keys: 'scores' and 'labels'")

    labels = _clean_labels(data["labels"])
    N = labels.shape[0]

    scores = data["scores"]
    scores_video = _scores_to_video_level(scores, N)  # (N, C)
    preds = np.argmax(scores_video, axis=1).astype(np.int64)

    return labels, preds


def confusion_matrix_numpy(labels, preds, n_classes):
    labels = np.array(labels).reshape(-1).astype(np.int64)
    preds = np.array(preds).reshape(-1).astype(np.int64)

    mask = (labels >= 0) & (labels < n_classes) & (preds >= 0) & (preds < n_classes)
    labels = labels[mask]
    preds = preds[mask]

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    return cm.astype(np.float32)


def plot_cm_blue(
    cm,
    class_names,
    out_prefix,
    normalize=True,
    percent=True,
    show_values=True,
    value_threshold=5.0,   # percent=True: >=5% 才标注
    fig_size=14,
    tick_font=7,
    axis_font=12,
    dpi_png=600,
    title=None,
):
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)

    cm_show = cm * 100.0 if percent else cm
    vmin, vmax = 0.0, (100.0 if percent else 1.0)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm_show, interpolation="nearest", cmap="Blues", vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel("Predicted label", fontsize=axis_font)
    ax.set_ylabel("True label", fontsize=axis_font)
    if title:
        ax.set_title(title, fontsize=axis_font + 1)

    n = len(class_names)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=tick_font)
    ax.set_yticklabels(class_names, fontsize=tick_font)
    ax.tick_params(axis="both", which="both", length=0)

    if show_values:
        thr = float(value_threshold)
        text_color_switch = cm_show.max() * 0.6 if cm_show.size else 0.0
        for i in range(n):
            for j in range(n):
                val = float(cm_show[i, j])
                if val < thr:
                    continue
                txt = f"{val:.1f}" if percent else f"{val:.2f}"
                ax.text(
                    j, i, txt,
                    ha="center", va="center",
                    fontsize=5,
                    color="white" if val >= text_color_switch else "black"
                )

    fig.tight_layout()
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out_prefix + ".pdf", bbox_inches="tight")          # 矢量：放大不糊
    fig.savefig(out_prefix + ".png", dpi=dpi_png, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(out_prefix + ".pdf")
    print(out_prefix + ".png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--num_classes", type=int, default=None)

    parser.add_argument("--no_normalize", action="store_true", help="disable row-normalization")
    parser.add_argument("--no_percent", action="store_true", help="show ratio (0-1) instead of percent")
    parser.add_argument("--no_values", action="store_true", help="do not annotate values")
    parser.add_argument("--value_threshold", type=float, default=5.0)
    parser.add_argument("--fig_size", type=float, default=14.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    class_names = build_class_names_from_list(args.train_list, num_classes=args.num_classes)
    labels, preds = load_npz_video_preds(args.npz)

    cm = confusion_matrix_numpy(labels, preds, n_classes=len(class_names))

    plot_cm_blue(
        cm=cm,
        class_names=class_names,
        out_prefix=args.out_prefix,
        normalize=(not args.no_normalize),
        percent=(not args.no_percent),
        show_values=(not args.no_values),
        value_threshold=args.value_threshold,
        fig_size=float(args.fig_size),
        dpi_png=int(args.dpi),
        title=args.title
    )


if __name__ == "__main__":
    main()
