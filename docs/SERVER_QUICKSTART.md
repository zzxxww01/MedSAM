# 服务器端操作指南

## 第一步：解压AMOS22数据集

登录服务器后执行：

```bash
cd ~/chengang/zxw/MedSAM
git pull
```

### 方法1：使用自动化脚本（推荐）
```bash
bash setup_amos22.sh
```

### 方法2：手动解压
```bash
cd data
unzip amos22.zip
ls -la  # 查看解压后的目录名
```

---

## 第二步：运行数据预处理

预处理脚本已支持自动检测目录名，直接运行即可：

```bash
cd ~/chengang/zxw/MedSAM
python pre_AMOS22.py
```

预期输出：
- 处理进度条
- 最终输出：`预处理完成! 输出目录: data/npy/CT_AMOS`

---

## 第三步：验证预处理结果

```bash
# 检查生成的文件数量
ls data/npy/CT_AMOS/imgs/ | wc -l
ls data/npy/CT_AMOS/gts/ | wc -l
```

预期：每个目录应该有数千到数万个 .npy 文件

---

## 第四步：启动Baseline训练 (A0)

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_ADDR=localhost \
MASTER_PORT=12355 \
MPLBACKEND=Agg \
python train_multi_gpus.py \
    -i data/npy/CT_AMOS \
    -task_name MedSAM-AMOS-A0-Baseline \
    -model_type vit_b \
    -checkpoint work_dir/medsam_vit_b.pth \
    -num_epochs 200 \
    -batch_size 2 \
    -lr 0.0001 \
    --world_size 2 \
    -use_amp
```

---

## 监控训练进度

### 查看GPU使用情况
```bash
watch -n 1 nvidia-smi
```

### 查看训练日志（另开一个终端）
```bash
tail -f work_dir/MedSAM-AMOS-A0-Baseline-*/train.log
```

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| 找不到数据集 | 确保已解压 amos22.zip |
| CUDA OOM | 减小 batch_size 到 1 |
| 端口占用 | 修改 MASTER_PORT 为其他值 |
| checkpoint不存在 | 检查 work_dir/medsam_vit_b.pth 是否存在 |

---

## 完整实验流程

详见 `docs/FULL_EXPERIMENT_PLAN.md`

