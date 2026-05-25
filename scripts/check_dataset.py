#!/usr/bin/env python3
"""
数据集健康检查工具
==================
在训练之前检查你的 YOLO 数据集是否存在常见问题，避免训练时报错。

使用方法:
    python scripts/check_dataset.py --data configs/data_kaggle.yaml
    python scripts/check_dataset.py --data configs/data_server.yaml

检查项目:
    1. data.yaml 文件格式是否正确
    2. 训练集/验证集图片和标签是否存在
    3. 标签文件格式是否正确（YOLO 格式: class_id x_center y_center width height）
    4. 类别 ID 是否在合法范围内
    5. 坐标是否归一化到 [0, 1]
    6. 是否存在空标签文件
    7. 图片和标签是否一一对应
"""

import argparse
import sys
from pathlib import Path

import yaml


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="YOLO 数据集健康检查工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="数据集配置文件路径（data.yaml）",
    )
    return parser.parse_args()


def load_yaml(yaml_path):
    """加载 YAML 文件"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


class DatasetChecker:
    """数据集检查器"""

    def __init__(self, yaml_path):
        self.yaml_path = Path(yaml_path)
        self.errors = []    # 严重错误（会导致训练失败）
        self.warnings = []  # 警告（可能影响效果）
        self.info = []      # 信息

    def check(self):
        """运行所有检查"""
        print("=" * 60)
        print("         YOLO 数据集健康检查")
        print("=" * 60)
        print(f"  配置文件: {self.yaml_path}")
        print()

        # 依次执行各项检查
        self._check_yaml()
        self._check_paths()
        self._check_labels()

        # 打印报告
        self._print_report()

        return len(self.errors) == 0

    def _check_yaml(self):
        """检查 YAML 文件格式"""
        print("[1/3] 检查 data.yaml 格式 ...")

        if not self.yaml_path.exists():
            self.errors.append(f"data.yaml 文件不存在: {self.yaml_path}")
            return

        try:
            data = load_yaml(self.yaml_path)
        except yaml.YAMLError as e:
            self.errors.append(f"data.yaml 格式错误: {e}")
            return

        # 检查必需字段
        required_fields = ["train", "val", "nc", "names"]
        for field in required_fields:
            if field not in data:
                self.errors.append(f"data.yaml 缺少必需字段: '{field}'")

        if "nc" in data and "names" in data:
            if data["nc"] != len(data["names"]):
                self.errors.append(
                    f"类别数量不匹配: nc={data['nc']}, 但 names 有 {len(data['names'])} 个"
                )

        self.info.append(f"类别数量 (nc): {data.get('nc', 'N/A')}")
        self.info.append(f"类别名称: {data.get('names', 'N/A')}")
        self.data = data

    def _check_paths(self):
        """检查图片和标签路径"""
        print("[2/3] 检查数据路径 ...")

        if self.errors:
            # 如果 YAML 检查就有错，跳过路径检查
            return

        data = self.data

        # 检查训练集路径
        train_img_dir = Path(data["train"])
        if train_img_dir.exists():
            train_images = self._get_image_files(train_img_dir)
            self.info.append(f"训练集图片数: {len(train_images)}")
        else:
            self.errors.append(f"训练集图片目录不存在: {train_img_dir}")
            train_images = []

        # 检查验证集路径
        val_img_dir = Path(data["val"])
        if val_img_dir.exists():
            val_images = self._get_image_files(val_img_dir)
            self.info.append(f"验证集图片数: {len(val_images)}")
        else:
            self.errors.append(f"验证集图片目录不存在: {val_img_dir}")
            val_images = []

        # 检查图片和标签是否一一对应
        for split, img_dir_path in [("训练集", train_img_dir), ("验证集", val_img_dir)]:
            if not img_dir_path.exists():
                continue

            label_dir = img_dir_path.parent / "labels"
            if not label_dir.exists():
                self.warnings.append(f"{split}标签目录不存在: {label_dir}")
                self.warnings.append(f"  请确认标签目录路径正确（通常和 images 同级的 labels/ 目录）")
                continue

            label_files = list(label_dir.glob("*.txt"))
            img_stems = {p.stem for p in self._get_image_files(img_dir_path)}
            label_stems = {p.stem for p in label_files}

            # 图片没有对应标签
            no_label = img_stems - label_stems
            if no_label:
                self.warnings.append(
                    f"{split}有 {len(no_label)} 张图片没有对应的标签文件"
                )
                if len(no_label) <= 5:
                    for stem in sorted(no_label):
                        self.warnings.append(f"  缺少标签: {stem}.txt")

            # 标签没有对应图片
            no_image = label_stems - img_stems
            if no_image:
                self.warnings.append(
                    f"{split}有 {len(no_image)} 个标签文件没有对应的图片"
                )
                if len(no_image) <= 5:
                    for stem in sorted(no_image):
                        self.warnings.append(f"  缺少图片: {stem}.*")

        self.train_img_dir = train_img_dir
        self.val_img_dir = val_img_dir

    def _check_labels(self):
        """检查标签文件内容"""
        print("[3/3] 检查标签内容 ...")

        if self.errors and not hasattr(self, "train_img_dir"):
            return

        data = self.data
        nc = data.get("nc", 0)
        total_labels = 0
        total_instances = 0
        empty_labels = 0
        bad_format = 0
        out_of_range_cls = 0
        bad_coords = 0

        # 统计每个类别的标签数量
        class_counts = {}

        for split_key in ["train", "val"]:
            img_dir = Path(data[split_key])
            label_dir = img_dir.parent / "labels"

            if not label_dir.exists():
                continue

            for label_file in label_dir.glob("*.txt"):
                total_labels += 1

                try:
                    content = label_file.read_text(encoding="utf-8").strip()

                    # 检查空标签
                    if not content:
                        empty_labels += 1
                        continue

                    # 逐行检查
                    for line_num, line in enumerate(content.split("\n"), 1):
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()

                        # YOLO 格式: class_id x_center y_center width height
                        if len(parts) != 5:
                            bad_format += 1
                            if bad_format <= 3:
                                self.errors.append(
                                    f"{label_file.name} 第 {line_num} 行格式错误: "
                                    f"'{line}'（应为 5 个数字）"
                                )
                            continue

                        try:
                            cls_id = int(parts[0])
                            coords = [float(x) for x in parts[1:]]
                        except ValueError:
                            bad_format += 1
                            continue

                        total_instances += 1

                        # 统计每个类别的实例数
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

                        # 检查类别 ID 范围
                        if cls_id < 0 or cls_id >= nc:
                            out_of_range_cls += 1
                            if out_of_range_cls <= 3:
                                self.warnings.append(
                                    f"{label_file.name} 第 {line_num} 行: "
                                    f"类别 ID {cls_id} 超出范围 [0, {nc - 1}]"
                                )

                        # 检查坐标归一化
                        for coord_name, coord_val in zip(
                            ["x_center", "y_center", "width", "height"], coords
                        ):
                            if not (0.0 <= coord_val <= 1.0):
                                bad_coords += 1
                                if bad_coords <= 3:
                                    self.warnings.append(
                                        f"{label_file.name} 第 {line_num} 行: "
                                        f"{coord_name}={coord_val} 不在 [0,1] 范围内"
                                    )
                                break

                except Exception as e:
                    self.errors.append(f"读取标签文件出错 {label_file.name}: {e}")

        # 汇总标签检查结果
        self.info.append(f"检查标签文件数: {total_labels}")
        self.info.append(f"标签实例总数: {total_instances}")

        # 打印每个类别的标签数量
        if class_counts:
            names = data.get("names", [])
            self.info.append("各类别标签数量:")
            for cls_id in range(nc):
                name = names[cls_id] if cls_id < len(names) else str(cls_id)
                count = class_counts.get(cls_id, 0)
                self.info.append(f"  类别 {cls_id} ({name}): {count} 个标签")

        if empty_labels > 0:
            self.warnings.append(f"发现 {empty_labels} 个空标签文件")
        if bad_format > 0:
            self.errors.append(f"发现 {bad_format} 行格式错误的标签")
        if out_of_range_cls > 0:
            self.errors.append(f"发现 {out_of_range_cls} 个超出范围的类别 ID")
        if bad_coords > 0:
            self.warnings.append(f"发现 {bad_coords} 个未归一化的坐标值")

    def _get_image_files(self, directory):
        """获取目录下的所有图片文件"""
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        return [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in extensions]

    def _print_report(self):
        """打印检查报告"""
        print()
        print("=" * 60)
        print("         检查报告")
        print("=" * 60)

        # 基本信息
        if self.info:
            print("\n📋 基本信息:")
            for msg in self.info:
                print(f"   {msg}")

        # 警告
        if self.warnings:
            print(f"\n⚠️  警告 ({len(self.warnings)} 项):")
            for msg in self.warnings:
                print(f"   {msg}")

        # 错误
        if self.errors:
            print(f"\n❌ 错误 ({len(self.errors)} 项):")
            for msg in self.errors:
                print(f"   {msg}")

        # 最终结论
        print()
        print("-" * 60)
        if not self.errors and not self.warnings:
            print("✅ 数据集检查通过！没有发现问题，可以开始训练。")
        elif not self.errors:
            print("⚠️  数据集基本可用，但有一些警告。建议检查后再训练。")
        else:
            print("❌ 数据集存在问题，请修复上面的错误后再训练！")
        print("=" * 60)


def main():
    args = parse_args()

    # 检查 PyYAML 是否安装
    try:
        import yaml  # noqa: F811
    except ImportError:
        print("错误: 需要安装 PyYAML 库")
        print("请运行: pip install pyyaml")
        sys.exit(1)

    checker = DatasetChecker(args.data)
    ok = checker.check()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
