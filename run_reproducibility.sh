#!/bin/bash
# Reproducibility 实验完整运行脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "FE-GIN Reproducibility 实验运行脚本"
echo "=========================================="

# 1. 检查数据集
echo ""
echo "步骤 1/3: 检查数据集..."
if [ ! -d "data/data_bil/force_load" ] || [ ! -d "data/data_exp/force_load" ] || [ ! -d "data/data_grf/force_load" ]; then
    echo "❌ 错误：源数据集不存在！"
    echo ""
    echo "请先下载数据集："
    echo "  git clone https://huggingface.co/datasets/zhugx/fe_gin_data data"
    echo ""
    echo "或从 Hugging Face 网页下载后解压到 data/ 目录"
    exit 1
fi
echo "✓ 源数据集存在"

# 2. 生成混合数据集
echo ""
echo "步骤 2/3: 生成混合数据集（train/val 分割）..."
if [ ! -d "data/data_mix/force_load/train" ]; then
    echo "正在生成数据集（使用 seed=42）..."
    cd igfe_unet/script
    python data_process.py
    cd ../..
    echo "✓ 数据集生成完成"
else
    echo "✓ 混合数据集已存在，跳过生成"
    echo "  如需重新生成，请删除: data/data_mix/force_load/"
fi

# 3. 运行 reproducibility 实验
echo ""
echo "步骤 3/3: 运行 reproducibility 实验..."
echo "  模型: MSE-M, LM-M, GM-M"
echo "  Seeds: 7, 17, 27, 37, 42, 47, 123, 2024, 2025, 3407"
echo "  输出: results/reproducibility/"
echo ""
read -p "是否开始运行？这可能需要几小时... (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd igfe_unet/script
    python reproducibility.py
    cd ../..
    echo ""
    echo "=========================================="
    echo "✓ 实验完成！"
    echo "=========================================="
    echo "输出位置: results/reproducibility/"
    echo "  - configs/: 配置文件和 protocol"
    echo "  - models/: 每个 seed 的模型权重"
    echo "  - metrics/: 统计结果 CSV"
    echo "  - figures/: 可视化图表"
else
    echo "已取消"
fi
