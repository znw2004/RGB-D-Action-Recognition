# 检测PyTorch版本和GPU可用性的脚本
import torch
import cv2
from PIL import Image


def check_pytorch_info():
    print("=" * 50)
    # 1. 检测PyTorch版本
    torch_version = torch.__version__
    print(f"当前PyTorch版本：{torch_version}")
    # 验证是否为1.10版本
    if "1.10" in torch_version:
        print("✅ 确认是PyTorch 1.10系列版本")
    else:
        print("❌ 不是PyTorch 1.10版本，请重新安装")

    # 2. 检测CUDA相关信息
    print("\n--- CUDA 信息 ---")
    # 检测PyTorch关联的CUDA版本
    cuda_version = torch.version.cuda
    print(f"PyTorch关联的CUDA版本：{cuda_version if cuda_version else '未关联CUDA（CPU版本）'}")
    # 检测GPU是否可用
    cuda_available = torch.cuda.is_available()
    print(f"GPU是否可用：{'✅ 可用' if cuda_available else '❌ 不可用'}")

    if cuda_available:
        # 输出可用GPU数量和名称
        gpu_count = torch.cuda.device_count()
        print(f"可用GPU数量：{gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i} 名称：{gpu_name}")
    print("=" * 50)

    # 验证opencv-python版本
    cv2_version = cv2.__version__
    print(f"opencv-python版本：{cv2_version}")
    if cv2_version == "4.5.1":  # opencv的__version__输出不带最后两位（4.5.1.48显示为4.5.1）
        print("✅ opencv-python 4.5.1.48 ")
    else:
        print("❌ opencv-python版本不符，请重新安装")

    # 验证Pillow版本
    pillow_version = Image.__version__
    print(f"\nPillow版本：{pillow_version}")
    if pillow_version == "8.3.1":
        print("✅ Pillow 8.3.1 ")
    else:
        print("❌ Pillow版本不符，请重新安装")
#pip3 uninstall torch torchvision torchaudio -y



if __name__ == "__main__":
    check_pytorch_info()