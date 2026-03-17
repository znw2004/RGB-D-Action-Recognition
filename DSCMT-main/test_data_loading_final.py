# test_data_loading_final.py
import sys
import os

sys.path.append('.')

import torch
from PIL import Image
import torchvision.transforms as transforms


def test_data_loading():
    print("最终数据加载测试")
    print("=" * 70)

    # 检查文件路径
    base_path = r"E:\transformer实验代码\DSCMT-main"
    train_list = os.path.join(base_path, "train_test_files", "ntu120_train_list.txt")

    if not os.path.exists(train_list):
        print(f"错误: 列表文件不存在 {train_list}")
        return False

    print(f"✅ 找到列表文件: {train_list}")

    # 检查文件内容
    with open(train_list, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"列表文件包含 {len(lines)} 行")
        if lines:
            print(f"第一行示例: {lines[0].strip()}")

    # 导入dataset
    try:
        from dataset import TSNDataSet
        print("✅ 成功导入 TSNDataSet")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

    # 创建简单的transform
    class SimpleTransform:
        def __init__(self):
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

        def __call__(self, images):
            # images是一个包含[RGB图像, 深度图像]的列表
            transformed = []
            for img in images:
                transformed.append(self.transform(img))
            return transformed

    # 创建数据集实例
    try:
        print("\n创建数据集实例...")
        dataset = TSNDataSet(
            root_path="",
            list_file=train_list,
            num_segments=3,
            new_length=1,
            modality='Appearance',
            image_tmpl='img_{:05d}.jpg',
            transform=SimpleTransform(),
            random_shift=True,
            test_mode=False
        )
        print(f"✅ 数据集创建成功")
        print(f"   数据集大小: {len(dataset)} 个样本")
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试加载单个样本
    try:
        print("\n测试加载第一个样本...")
        data, label = dataset[0]

        print(f"✅ 样本加载成功!")
        print(f"   返回数据长度: {len(data)} 个tensor")
        print(f"   第一个tensor形状: {data[0].shape}")
        print(f"   数据类型: {type(data[0])}")
        print(f"   标签: {label}")

        # 验证图像值范围
        print(f"\n验证图像值范围:")
        for i, img_tensor in enumerate(data):
            print(f"   图像{i}: min={img_tensor.min():.3f}, max={img_tensor.max():.3f}, "
                  f"mean={img_tensor.mean():.3f}, shape={img_tensor.shape}")

        # 测试第二个样本
        print(f"\n测试第二个样本...")
        data2, label2 = dataset[1]
        print(f"✅ 第二个样本加载成功!")
        print(f"   标签: {label2}")

    except Exception as e:
        print(f"❌ 样本加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 创建DataLoader测试批量加载
    try:
        print(f"\n创建DataLoader测试批量加载...")
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=0
        )

        print(f"✅ DataLoader创建成功")

        # 获取一个批次
        for i, (batch_data, batch_labels) in enumerate(dataloader):
            print(f"\n批次 {i}:")
            print(f"   数据tensor数量: {len(batch_data)}")
            print(f"   每个tensor形状: {batch_data[0].shape}")
            print(f"   标签: {batch_labels.tolist()}")
            print(f"   标签形状: {batch_labels.shape}")

            # 验证批处理是否正确
            if len(batch_data) == 2:  # 应该是2个图像（RGB + 深度）
                print(f"   批处理验证通过!")
            else:
                print(f"   ⚠️  警告: 期望2个tensor，得到{len(batch_data)}个")

            if i >= 1:  # 只看前2个批次
                break

    except Exception as e:
        print(f"❌ DataLoader测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试不同模态
    try:
        print(f"\n测试不同模态...")

        # 测试Motion模态（如果需要）
        print(f"当前仅测试Appearance模态，Motion模态需要动态图像数据")

    except Exception as e:
        print(f"❌ 模态测试失败: {e}")

    print(f"\n" + "=" * 70)
    print(f"🎉 所有测试通过!")
    print(f"\n下一步:")
    print(f"1. 可以开始训练模型")
    print(f"2. 使用命令: python main.py ntu120 Appearance {train_list} \\")
    print(f"     --arch resnet50 --num_segment 1 --lr 0.001 --lr_steps 25 50 \\")
    print(f"     --epochs 5 -b 16 --snapshot_pref test_model \\")
    print(f"     --val_list {os.path.join(base_path, 'train_test_files', 'ntu120_test_list.txt')}")
    print(f"\n注意: 建议先用少量epochs和batch size测试")

    return True


def quick_path_check():
    """快速检查关键路径"""
    print("快速路径检查")
    print("-" * 50)

    base_path = r"E:\transformer实验代码\DSCMT-main"

    # 检查关键文件
    paths_to_check = [
        ("项目根目录", base_path),
        ("dataset.py", os.path.join(base_path, "dataset.py")),
        ("训练列表", os.path.join(base_path, "train_test_files", "ntu120_train_list.txt")),
        ("测试列表", os.path.join(base_path, "train_test_files", "ntu120_test_list.txt")),
        ("RGB数据", os.path.join(base_path, "data", "RGB")),
        ("深度数据", os.path.join(base_path, "data", "Depth")),
    ]

    all_good = True
    for name, path in paths_to_check:
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        if not exists:
            all_good = False

    return all_good


if __name__ == "__main__":
    print("开始数据加载测试")
    print("=" * 70)

    # 先检查路径
    if not quick_path_check():
        print("\n❌ 路径检查失败，请检查上述路径")
        sys.exit(1)

    print("\n" + "=" * 70)

    # 运行主要测试
    success = test_data_loading()

    if success:
        print("\n✅ 所有测试完成，可以开始训练!")
    else:
        print("\n❌ 测试失败，请检查错误信息")
        sys.exit(1)