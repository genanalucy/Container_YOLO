#!/usr/bin/env python3
"""
集装箱检测推理预测脚本
======================
使用训练好的 YOLO 模型对图片或视频进行集装箱检测。

使用方法:
    # 检测单张图片
    python scripts/predict.py --model runs/exp/weights/best.pt --source test.jpg

    # 检测整个文件夹
    python scripts/predict.py --model runs/exp/weights/best.pt --source test_images/

    # 调整置信度阈值
    python scripts/predict.py --model best.pt --source test.jpg --conf 0.5
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="集装箱检测 YOLO 推理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/predict.py --model runs/exp/weights/best.pt --source test.jpg
  python scripts/predict.py --model best.pt --source test_images/ --conf 0.5
  python scripts/predict.py --model best.pt --source video.mp4 --name my_predict
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="训练好的模型权重路径（必填）。例如: runs/exp/weights/best.pt",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="要检测的图片路径、文件夹路径或视频路径（必填）。",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值（默认: 0.25）。只显示置信度高于此值的检测结果。"
        "降低会检测更多目标（可能有误检），提高会减少目标（只保留高置信度的）。",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs",
        help="结果保存的根目录（默认: runs）。",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="predict",
        help="本次预测的实验名称（默认: predict）。结果保存到 project/name/ 下。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推理设备（默认: 0，即第 1 块 GPU）。CPU 推理: 'cpu'。",
    )

    return parser.parse_args()


def print_banner(args):
    """打印推理配置信息"""
    print("=" * 60)
    print("         集装箱检测 — YOLO 推理预测")
    print("=" * 60)
    print(f"  模型权重   : {args.model}")
    print(f"  输入来源   : {args.source}")
    print(f"  置信度阈值 : {args.conf}")
    print(f"  保存路径   : {args.project}/{args.name}/")
    print(f"  推理设备   : {args.device}")
    print("=" * 60)
    print()


def print_results_summary(results_list):
    """打印检测结果摘要"""
    print("\n" + "=" * 60)
    print("         检测结果摘要")
    print("=" * 60)

    total_images = len(results_list)
    total_boxes = 0
    class_counts = {}

    for result in results_list:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            num_boxes = len(boxes)
            total_boxes += num_boxes

            # 统计每个类别的检测数量
            for cls_id in boxes.cls.cpu().numpy():
                cls_name = result.names[int(cls_id)]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    # 打印统计信息
    print(f"  处理图片数     : {total_images}")
    print(f"  检测到的目标数 : {total_boxes}")
    print(f"  平均每张图     : {total_boxes / max(total_images, 1):.1f} 个目标")

    if class_counts:
        print(f"\n  各类别统计:")
        # 类别名映射（原始数据集用的是数字字符串）
        name_map = {"0": "集装箱-类型0", "1": "集装箱-类型1", "2": "集装箱-类型2", "3": "集装箱-类型3"}
        for cls_name, count in sorted(class_counts.items()):
            display_name = name_map.get(cls_name, cls_name)
            print(f"    {display_name}: {count} 个")

    # 打印保存路径
    if results_list:
        save_dir = results_list[0].save_dir
        print(f"\n  结果保存位置   : {save_dir}")
        print(f"  带标注的图片在 : {save_dir}/")

    print("=" * 60)


def main():
    """主推理流程"""
    # 1. 解析参数
    args = parse_args()

    # 2. 检查模型文件是否存在
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 找不到模型文件 '{args.model}'")
        print("请先完成训练，或指定正确的权重路径。")
        sys.exit(1)

    # 3. 检查输入来源是否存在
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"错误: 找不到输入来源 '{args.source}'")
        sys.exit(1)

    # 4. 打印配置
    print_banner(args)

    # 5. 加载模型
    print(f"[1/2] 正在加载模型: {args.model} ...")
    model = YOLO(args.model)
    print(f"      模型加载完成！")

    # 6. 运行推理
    print(f"[2/2] 正在检测: {args.source} ...")

    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,           # 保存带标注的图片
        project=args.project,
        name=args.name,
        device=args.device,
    )

    # 7. 打印结果摘要
    print_results_summary(results)


if __name__ == "__main__":
    main()
