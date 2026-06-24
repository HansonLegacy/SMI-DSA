import os
import argparse
import cv2
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

def process_single_image(args):
    """
    单个图片处理函数，将被多进程调用
    """
    src_file, input_root, output_root, width, height = args
    
    try:
        # 1. 计算相对路径和目标路径
        relative_path = src_file.relative_to(input_root)
        dst_file = output_root / relative_path
        
        # 2. 检查目标文件是否已存在（可选，跳过已处理的）
        # if dst_file.exists():
        #     return True

        # 3. 确保父目录存在 (exist_ok=True 是线程/进程安全的)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 4. 读取图片
        img = cv2.imread(str(src_file))
        if img is None:
            return False # 读取失败
            
        # 5. Resize
        resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        
        # 6. 保存
        cv2.imwrite(str(dst_file), resized_img)
        return True
        
    except Exception as e:
        # 捕获错误，防止单个文件崩溃导致进程挂掉
        return False

def resize_images_multiprocess(input_root, height, width, output_root=None, num_workers=None):
    input_path = Path(input_root).resolve()
    
    # --- 1. 准备路径 ---
    original_dataset_name = input_path.name
    if height == width:
        size_suffix = str(height)
    else:
        size_suffix = f"{height}x{width}"
    new_dataset_name = f"{original_dataset_name}_resize_{size_suffix}"
    
    if output_root:
        base_output_dir = Path(output_root).resolve()
        output_path = base_output_dir / new_dataset_name
    else:
        output_path = input_path.parent / new_dataset_name

    print(f"========================================")
    print(f"原数据集: {input_path}")
    print(f"输出位置: {output_path}")
    print(f"目标尺寸: {height}x{width}")
    
    # --- 2. 扫描文件 ---
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    image_files = []
    
    print("正在扫描文件列表 (这是单线程操作，请稍候)...")
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if Path(file).suffix.lower() in valid_extensions:
                image_files.append(Path(root) / file)
    
    total_files = len(image_files)
    if total_files == 0:
        print("未找到图片。")
        return

    # --- 3. 准备多进程任务参数 ---
    # 确定进程数，默认使用 (CPU核数 - 4) 以避免卡死系统，或者用户指定
    if num_workers is None:
        num_workers = max(1, cpu_count() - 4) 
    
    print(f"共发现 {total_files} 张图片。")
    print(f"启动 {num_workers} 个进程进行并行处理...")
    print(f"========================================")

    # 打包参数：每个任务都需要知道源路径、根目录、目标根目录、尺寸
    # 使用 list comprehension 生成任务列表
    tasks = [(f, input_path, output_path, width, height) for f in image_files]

    # --- 4. 开始多进程处理 ---
    start_time = time.time()
    
    # 使用 imap_unordered 可以让进度条更流畅，哪个处理完就显示哪个
    with Pool(processes=num_workers) as pool:
        # results 是一个迭代器
        results = list(tqdm(pool.imap_unordered(process_single_image, tasks), total=total_files, unit="img"))

    end_time = time.time()
    duration = end_time - start_time
    
    # --- 5. 统计结果 ---
    success_count = sum(results)
    fail_count = total_files - success_count
    
    print("\n--------------------------------")
    print(f"处理完成！耗时: {duration:.2f} 秒")
    print(f"平均速度: {total_files/duration:.2f} it/s")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    if fail_count > 0:
        print("（失败通常是因为图片文件损坏无法读取）")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多进程快速Resize数据集")
    
    parser.add_argument('--data_path', type=str, required=True, help='原始数据集路径')
    parser.add_argument('--height', type=int, required=True, help='目标高度')
    parser.add_argument('--width', type=int, required=True, help='目标宽度')
    parser.add_argument('--output_root', type=str, default=None, help='指定输出根目录')
    parser.add_argument('--workers', type=int, default=None, help='指定进程数 (默认使用CPU核数-4)')

    args = parser.parse_args()

    resize_images_multiprocess(args.data_path, args.height, args.width, args.output_root, args.workers)