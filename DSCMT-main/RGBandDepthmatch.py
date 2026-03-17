# verify_depth_structure.py
import os

# 检查一个具体的深度文件夹
depth_folder = r"E:\transformer实验代码\DSCMT-main\data\Depth\S016C001P007R001A001"

print(f"检查深度文件夹: {depth_folder}")
print("=" * 60)

if os.path.exists(depth_folder):
    files = os.listdir(depth_folder)
    print(f"文件夹中存在 {len(files)} 个文件/文件夹")

    if len(files) > 0:
        print("\n前10个项目:")
        for i, item in enumerate(files[:10]):
            full_path = os.path.join(depth_folder, item)
            if os.path.isfile(full_path):
                size = os.path.getsize(full_path)
                print(f"  {i + 1}. {item} ({size} 字节)")
            else:
                print(f"  {i + 1}. {item}/ (子文件夹)")
else:
    print("文件夹不存在!")