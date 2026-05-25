#!/usr/bin/env python3
"""
集装箱检测模型训练脚本
======================
使用 YOLO11 训练集装箱（Container）目标检测模型。

使用方法:
    # 使用默认参数训练（yolo11n，50 轮）
    python scripts/train.py --data configs/data_kaggle.yaml

    # 自定义模型和参数
    python scripts/train.py --model yolo11s.pt --data configs/data_server.yaml --epochs 100 --batch 32

    # 在 CPU 上训练（不推荐，非常慢）
    python scripts/train.py --data configs/data_kaggle.yaml --device cpu
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="集装箱检测 YOLO 训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/train.py --data configs/data_kaggle.yaml
  python scripts/train.py --model yolo11s.pt --data configs/data_server.yaml --epochs 100
        """,
    )

    # ---- 模型相关 ----
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="模型文件名或路径（默认: yolo11n.pt）。"
        "可选: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt，"
        "或自定义 .pt 权重路径。",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="数据集配置文件路径（必填）。例如: configs/data_kaggle.yaml",
    )

    # ---- 训练参数 ----
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练轮数（默认: 50）。数据量小时可以设大一些，如 100。",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图像尺寸（默认: 640）。越大越准但越慢，可选 320/416/640/1280。",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="批大小（默认: 16）。显存不够时调小，如 8 或 4。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="训练设备（默认: 0，即第 1 块 GPU）。"
        "多卡: '0,1'；CPU: 'cpu'。",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="早停耐心值（默认: 20）。连续多少轮指标没有提升就停止训练。",
    )

    # ---- 输出相关 ----
    parser.add_argument(
        "--project",
        type=str,
        default="runs",
        help="保存目录的根路径（默认: runs）。",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="本次实验名称（默认: exp）。结果会保存到 project/name/ 下。",
    )

    return parser.parse_args()


def print_banner(args):
    """打印训练配置信息"""
    print("=" * 60)
    print("         集装箱检测 — YOLO 训练")
    print("=" * 60)
    print(f"  模型文件   : {args.model}")
    print(f"  数据集配置 : {args.data}")
    print(f"  训练轮数   : {args.epochs}")
    print(f"  图像尺寸   : {args.imgsz}")
    print(f"  批大小     : {args.batch}")
    print(f"  设备       : {args.device}")
    print(f"  早停耐心   : {args.patience}")
    print(f"  保存路径   : {args.project}/{args.name}/")
    print("=" * 60)
    print()


def print_results(results):
    """训练结束后打印结果摘要"""
    print("\n" + "=" * 60)
    print("         训练完成！结果摘要")
    print("=" * 60)

    # 获取最佳模型的指标
    try:
        # results 是 ultralytics 的 Results 对象
        metrics = results.results_dict if hasattr(results, "results_dict") else {}

        # 尝试打印关键指标
        if metrics:
            print(f"  精确率 (P)       : {metrics.get('metrics/precision(B)', 'N/A')}")
            print(f"  召回率 (R)       : {metrics.get('metrics/recall(B)', 'N/A')}")
            print(f"  mAP50            : {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"  mAP50-95         : {metrics.get('metrics/mAP50-95(B)', 'N/A')}")

        # 健壮性 fitness 指标
        fitness = metrics.get("fitness", "N/A")
        print(f"  综合适应度       : {fitness}")
    except Exception:
        print("  （指标详情请查看 runs/ 目录下的训练日志）")

    # 打印保存路径
    save_dir = results.save_dir if hasattr(results, "save_dir") else "runs/"
    print(f"\n  结果保存目录     : {save_dir}")
    print(f"  最佳权重         : {save_dir}/weights/best.pt")
    print(f"  最末权重         : {save_dir}/weights/last.pt")
    print("=" * 60)
    print("\n下一步: 用 best.pt 做推理预测")
    print(f"  python scripts/predict.py --model {save_dir}/weights/best.pt --source 你的图片路径.jpg")


def main():
    """主训练流程"""
    # 1. 解析参数
    args = parse_args()

    # 2. 检查数据集配置文件是否存在
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"错误: 找不到数据集配置文件 '{args.data}'")
        print("请先准备好 data.yaml 文件，参考 configs/ 目录下的模板。")
        sys.exit(1)

    # 3. 打印配置
    print_banner(args)

    # 4. 加载模型
    print(f"[1/3] 正在加载模型: {args.model} ...")
    model = YOLO(args.model)
    print(f"      模型加载完成！")

    # 5. 开始训练
    print(f"[2/3] 开始训练...")
    print(f"      提示: 训练过程中会在终端实时显示 loss 和 mAP 指标。")
    print(f"      如果指标连续 {args.patience} 轮没有提升，会自动早停。")
    print()

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )

    # 6. 打印结果
    print(f"[3/3] 训练结束。")
    print_results(results)


if __name__ == "__main__":
    main()
