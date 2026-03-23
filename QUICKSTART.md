# MemEvolve 评测快速入门 🚀

## 一键运行评测

```bash
cd /home/MemEvolve/Flash-Searcher-main
./eval.sh
```

就这么简单！脚本会自动：
- ✅ 检查环境配置
- ✅ 运行 3 个 GAIA 任务
- ✅ 生成完整的轨迹分析
- ✅ 输出任务正确率统计

## 三种使用方式

### 🔹 方式 1: 默认配置（最简单）
```bash
./eval.sh
```

### 🔹 方式 2: 自定义配置（推荐）
```bash
# 1. 修改配置文件
nano evaluation_config.sh

# 2. 运行评测
./eval.sh
```

**常用配置项：**
```bash
export EVAL_NUM_SAMPLE=10      # 评测任务数量
export EVAL_MAX_ROUNDS=1       # 演化轮数
export EVAL_BACKUP_RESULTS=true  # 自动备份旧结果
```

### 🔹 方式 3: 直接修改脚本
```bash
# 修改主脚本
nano run_evaluation.sh

# 找到配置部分并修改
# NUM_SAMPLE=3
# MAX_ROUNDS=1

# 运行
./run_evaluation.sh
```

## 查看结果

评测完成后，结果保存在 `evolve_demo_run/round_00/` 目录：

```bash
# 📁 任务轨迹（包含完整的 agent_trajectory）
cat evolve_demo_run/round_00/base_logs/1.json | jq .

# 📊 评测结果（答案和评判）
cat evolve_demo_run/round_00/result.jsonl | jq .

# 📈 分析报告（记忆操作分析）
cat evolve_demo_run/round_00/analysis_report.json | jq .

# 📉 正确率统计
grep -o '"judgement": "[^"]*"' evolve_demo_run/round_00/result.jsonl | sort | uniq -c
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `eval.sh` | 🚀 一键启动脚本（推荐使用） |
| `run_evaluation.sh` | 📋 主评测脚本（完整流程） |
| `evaluation_config.sh` | ⚙️ 配置文件（修改参数） |
| `README_EVALUATION.md` | 📖 完整使用文档 |

## 常见使用场景

### 场景 1: 快速测试（3个任务，约5分钟）
```bash
# 使用默认配置
./eval.sh
```

### 场景 2: 标准评测（10个任务，约15分钟）
```bash
# 修改 evaluation_config.sh
export EVAL_NUM_SAMPLE=10

# 运行
./eval.sh
```

### 场景 3: 完整演化（包含记忆系统演化）
```bash
# 修改 evaluation_config.sh
export EVAL_MAX_ROUNDS=3
export EVAL_MODE="full"

# 运行
./eval.sh
```

### 场景 4: 使用自定义数据集
```bash
# 1. 准备数据（GAIA格式）
mkdir -p data/custom_dataset
# 创建 metadata.jsonl

# 2. 修改 evaluation_config.sh
export EVAL_DATA_DIR="./data/custom_dataset"

# 3. 运行
./eval.sh
```

## 输出示例

运行后会看到类似输出：

```
========================================
MemEvolve GAIA 评测开始
========================================
→ 检查 Conda 环境: dl
✓ Conda 环境检查通过
→ 检查数据目录: ./data/gaia/validation
✓ 数据目录检查通过
→ 检查环境配置文件
✓ 环境配置检查通过

========================================
评测配置
========================================
数据目录: ./data/gaia/validation
保存目录: ./evolve_demo_run
任务数量: 3
演化轮数: 1
评测模式: base

========================================
开始运行评测
========================================
→ 这可能需要几分钟时间，请耐心等待...

[运行中...]

========================================
评测完成，分析结果
========================================
✓ 生成任务轨迹文件: 3 个
✓ 任务正确率: 3/3 = 100.00%
✓ 分析报告已生成: analysis_report.json (76K)
```

## 故障排查

如果遇到问题，检查以下项目：

1. **Conda 环境**
   ```bash
   conda env list  # 确认 'dl' 环境存在
   ```

2. **API 配置**
   ```bash
   cat .env  # 检查 API keys 配置
   ```

3. **数据文件**
   ```bash
   ls -la data/gaia/validation/metadata.jsonl
   ```

4. **清理重试**
   ```bash
   rm -rf evolve_demo_run
   ./eval.sh
   ```

## 进阶技巧

### 并行运行多个评测
```bash
# 评测1: 标准配置
./eval.sh

# 评测2: 修改保存目录后再次运行
export EVAL_SAVE_DIR="./evolve_run_experiment2"
./run_evaluation.sh
```

### 批量评测
```bash
for rounds in 1 2 3; do
    export EVAL_MAX_ROUNDS=$rounds
    export EVAL_SAVE_DIR="./evolve_run_rounds_${rounds}"
    ./run_evaluation.sh
done
```

### 结果对比
```bash
# 对比不同轮次的正确率
for dir in evolve_run_rounds_*; do
    echo "=== $dir ==="
    grep -o '"judgement": "[^"]*"' $dir/round_00/result.jsonl | sort | uniq -c
done
```

## 获取帮助

- 📖 完整文档: `README_EVALUATION.md`
- ⚙️ 配置说明: `evaluation_config.sh`
- 🔍 脚本源码: `run_evaluation.sh`

---

**提示**: 首次运行建议使用默认配置（3个任务），验证环境正确后再增加任务数量。
