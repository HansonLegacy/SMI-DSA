import os
import argparse
import cv2
from tqdm import tqdm
from pathlib import Path

def resize_images(input_root, height, width):
    """
    读取数据集，Resize图片，自动在同级目录下生成规范命名的输出文件夹。
    """
    input_path = Path(input_root).resolve() # 获取绝对路径，处理 ../ 等情况
    
    # 1. 自动构建输出文件夹名称和路径
    # 获取原文件夹名称 (例如: brain_data)
    original_dataset_name = input_path.name
    
    # 根据长宽是否相等，决定后缀是 _512 还是 _512x256
    if height == width:
        size_suffix = str(height)
    else:
        size_suffix = f"{height}x{width}"
        
    # 构建新名称: 原名_resize_大小 (例如: brain_data_resize_512)
    new_dataset_name = f"{original_dataset_name}_resize_{size_suffix}"
    
    # 输出路径在原路径的父目录下 (即与原数据集并列)
    output_path = input_path.parent / new_dataset_name
    
    print(f"========================================")
    print(f"原数据集: {input_path}")
    print(f"输出位置: {output_path}")
    print(f"目标尺寸: {height}x{width}")
    print(f"========================================")

    # 2. 扫描文件
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    image_files = []
    
    print("正在扫描文件...")
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if Path(file).suffix.lower() in valid_extensions:
                full_path = Path(root) / file
                image_files.append(full_path)
    
    if not image_files:
        print("未找到图片，请检查路径。")
        return

    print(f"共发现 {len(image_files)} 张图片，开始处理...")

    # 3. 遍历并处理
    for src_file in tqdm(image_files, desc="Processing"):
        # 计算相对路径，保持内部目录结构
        # 例如: src/data/A/1.png relative_to src/data -> A/1.png
        relative_path = src_file.relative_to(input_path)
        
        # 拼接目标路径
        dst_file = output_path / relative_path
        
        # 确保目标文件的父文件夹存在
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 读取与Resize
        img = cv2.imread(str(src_file))
        if img is None:
            continue # 跳过损坏文件
            
        # cv2.resize 接收 (width, height)
        resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        
        # 保存
        cv2.imwrite(str(dst_file), resized_img)

    print("\n所有图片处理完成！")
    print(f"新数据集已保存在: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动命名并Resize数据集")
    
    # 只需要输入原路径和尺寸
    parser.add_argument('--data_path', type=str, required=True, help='原始数据集文件夹路径')
    parser.add_argument('--height', type=int, required=True, help='目标高度')
    parser.add_argument('--width', type=int, required=True, help='目标宽度')

    args = parser.parse_args()

    resize_images(args.data_path, args.height, args.width)