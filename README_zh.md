# EpiCataNexus

**连接酶动力学预测与面向酶工程虚拟筛选的底物引导、口袋感知模型。**

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C">
  <img alt="Task" src="https://img.shields.io/badge/task-enzyme%20kinetics-2E8B57">
  <img alt="Status" src="https://img.shields.io/badge/release-research%20preview-6A5ACD">
</p>

> 当前状态：配套尚未正式发表论文的研究预览版。权重/数据地址、论文标识符和
> 最终软件许可证将在相关信息确定后补充。

[English README](README.md) · [安装](docs/INSTALL.md) ·
[数据说明](docs/DATA.md) · [复现指南](docs/REPRODUCIBILITY.md) ·
[权重说明](docs/WEIGHTS.md) · [模型卡](MODEL_CARD.md) · [参与贡献](CONTRIBUTING.md)

<p align="center">
  <img src="assets/figures/web/Fig1.webp" width="920" alt="EpiCataNexus 总体流程">
</p>

EpiCataNexus 将底物化学信息视为对蛋白质上下文的条件查询。模型联合使用
ProtT5–ESM-2 残基级交叉注意力、几何预测口袋 EGNN、PST 结构特征、
SMILES-Mamba 和 TRFM，并通过底物引导门控网络 SGGN 在预测前动态调整
蛋白侧特征。

主实现采用残基级 ProtT5/ESM-2 表征。另行发布的 pooled-feature checkpoint
通过 [`epicatanexus.legacy_pooled`](epicatanexus/legacy_pooled/) 兼容模块加载；
它们使用每个蛋白一个 ProtT5 向量和一个 ESM-2 向量，不能直接加载到残基级
`EpiCataNexus` 类中。

## 核心结果

本仓库提供论文所述 EpiCataNexus 架构的公开实现。下列数值为论文报告结果。
首批外部权重仅包含 `kcat` 与 `Km` pooled-feature 神经网络 checkpoint，
不发布任务相关的 XGBoost 权重。

| 任务 | 测试样本 | R² | RMSE | MAE | PCC |
|---|---:|---:|---:|---:|---:|
| `log10(kcat)` | 1,187 | 0.6470 | 0.7769 | 0.5618 | 0.8049 |
| `log10(Km)` | 1,984 | 0.6802 | 0.7253 | 0.5228 | 0.8253 |

工程筛选结果包括：

- P25910 实验突变方向预测正确 7/8。
- PETase 三个结构模板之间的排序 Spearman 相关系数均高于 0.951。
- TEM-1 双突变正上位性筛选 `NDCG@10 = 0.7716`。
- 综合候选池底物恢复 `Hit@5 = 0.5062`、`AUROC = 0.8309`。

<p align="center">
  <img src="assets/figures/web/Fig7.webp" width="920" alt="EpiCataNexus 详细架构">
</p>

## Pooled-feature 权重兼容

两份外部 checkpoint 使用相同的约 1550 万参数神经网络，包含口袋 EGNN、
pooled ProtT5/ESM-2 融合、SMILES-Mamba、TRFM/PST 投影、SGGN 和任务回归头。

| 任务 | 发布文件 | 历史来源文件名 | 公开位置 |
|---|---|---|---|
| `kcat` | `epicatanexus_kcat_pooled.safetensors` | `best_pocket_mamba_1792_clean_sggn.pkl` | [Hugging Face](https://huggingface.co/nnnnnnnnnnnn1111/EpiCataNexus/blob/7c78581fd5150a3bd60b91d158daafa8a7590133/epicatanexus_kcat_pooled.safetensors) |
| `Km` | `epicatanexus_km_pooled.safetensors` | `best_pocket_mamba_1792_km_sggn.pkl` | [Hugging Face](https://huggingface.co/nnnnnnnnnnnn1111/EpiCataNexus/blob/7c78581fd5150a3bd60b91d158daafa8a7590133/epicatanexus_km_pooled.safetensors) |

对可信权重执行严格结构校验：

```bash
python scripts/verify_legacy_checkpoint.py /path/to/checkpoint.pkl
```

输入格式、适用范围和限制见 [`docs/WEIGHTS.md`](docs/WEIGHTS.md) 与
[`MODEL_CARD.md`](MODEL_CARD.md)。

## 安装与轻量验证

```bash
conda env create -f environment.yml
conda activate epicatanexus
pip install -e .

python scripts/smoke_test.py
pytest
```

完整模型还需要与 PyTorch/CUDA 版本匹配的 `mamba-ssm`。详细说明见
[`docs/INSTALL.md`](docs/INSTALL.md)。可生成不含真实研究数据的合成输入用于检查接口：

```bash
python scripts/create_example_batch.py --output outputs/example_batches.pt
```

## 数据划分

```bash
python scripts/prepare_data.py \
  --input data/processed/kcat_manifest.tsv \
  --output-dir data/splits/kcat \
  --seed 3407
```

对 11,869 条有效记录，脚本生成 9,613 条训练、1,069 条验证和 1,187 条
独立测试记录。

## 训练、评估与预测

```bash
python scripts/train.py --config configs/kcat.yaml --device cuda

python scripts/evaluate.py \
  --checkpoint checkpoints/epicatanexus_kcat.pt \
  --batches data/features/kcat_test_batches.pt \
  --output-dir outputs/kcat_test

python scripts/predict.py \
  --checkpoint checkpoints/epicatanexus_kcat.pt \
  --batches data/features/new_pairs.pt \
  --output outputs/new_pair_predictions.csv
```

当前公开接口使用预处理后的张量批次。直接从序列和 SMILES 开始还需要结构
获取、fpocket、ProtT5、ESM-2、PST 和 TRFM 预处理；在模型权重及相关许可
确定前，仓库不会将其宣传为已经完成的一键功能。

## 论文真实配图

仓库内 `Fig1.png`—`Fig7.png` 全部来自作者提供的 `论文配图.rar`，没有重绘
或更改科学内容。README 使用由这些原图生成的轻量 WebP 副本；原始 PNG 的校验值见
[`assets/figures/SHA256SUMS`](assets/figures/SHA256SUMS)。

## 发布前状态

当前仓库不包含大型模型、预计算特征、原始数据库或本机安装包。许可证选择、
数据与权重托管地址、外部文件校验值以及论文 Data and Code Availability 声明
将在论文和外部资源准备完成后更新；当前研究预览版会明确保留这些待办状态。具体见
[`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md)。

