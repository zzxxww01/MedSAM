# 本周执行面板（实验 + 论文联动）

> 周期：2026-02-14 ~ 2026-02-21  
> 主线：先完成 A3R1，再决定 R2/R3 或进入 Attention

---

## 1. 本周目标

1. 完成 A3R1 训练与评估回填。
2. 输出 A0/A1/A2/A3/A3R1 同口径对比表。
3. 完成论文第3章初稿（含失败机制解释与修正策略）。

---

## 2. 已完成

- [x] A0 同口径评估回填。
- [x] A1/A2/A3 同口径评估回填。
- [x] A3R1 训练任务已启动（nohup + PID）。
- [x] 建立主控文档：`docs/THESIS_MASTER_GUIDE.md`
- [x] 建立技术报告：`docs/THESIS_TECHNICAL_REPORT.md`

---

## 3. 进行中

- [ ] A3R1 训练监控（日志/GPU/进程）。
- [ ] A3R1 评估执行与 `A3R1_summary.json` 生成。
- [ ] 第3章写作（方法、实验、机制解释）。

---

## 4. 待决策事项（依赖 A3R1 结果）

1. 若 A3R1 接近或超过 A2：直接进入 Attention。
2. 若 A3R1 仅部分恢复：执行 A3R2/A3R3。
3. 若 A3R1 仍明显弱于 A2：以 A2 作为后续主干。

---

## 5. 每日检查命令

```bash
cd ~/chengang/zxw/MedSAM
conda activate medsam
ps -fp $(cat work_dir/exp_logs/A3R1_train.pid)
tail -n 30 work_dir/exp_logs/A3R1_train.log
nvidia-smi
```

---

## 6. A3R1 完成后的立即动作

1. 后台评估：
```bash
mkdir -p work_dir/eval_metrics/logs
PY=/home/chengang/anaconda3/envs/medsam/bin/python
A3R1_CKPT=$(ls -dt work_dir/MedSAM-FLARE22-A3R1-Balance-a0.5-b1.0-s70-*/medsam_model_best.pth | head -n1)

nohup $PY eval_medsam_npz.py \
  --data_root data/npy/CT_Abd \
  --checkpoint "$A3R1_CKPT" \
  --exp_name A3R1 \
  --out_csv work_dir/eval_metrics/A3R1_case_metrics.csv \
  --out_json work_dir/eval_metrics/A3R1_summary.json \
  > work_dir/eval_metrics/logs/A3R1_eval.log 2>&1 < /dev/null &
echo $! > work_dir/eval_metrics/logs/A3R1_eval.pid
```

2. 汇总查看：
```bash
python - <<'PY'
import json
files = [
  "work_dir/eval_metrics/A0_summary.json",
  "work_dir/eval_metrics/A1_summary.json",
  "work_dir/eval_metrics/A2_summary.json",
  "work_dir/eval_metrics/A3_summary.json",
  "work_dir/eval_metrics/A3R1_summary.json",
]
for p in files:
    d = json.load(open(p, "r", encoding="utf-8"))
    print(f"{d['exp_name']}: DSC={d['dice_mean']:.6f}, HD95={d['hd95_mean']:.6f}, ASD={d['asd_mean']:.6f}")
PY
```

---

## 7. 论文写作联动任务

- [ ] 第3章 3.2/3.3 节：Inter/Intra 公式与动机补齐。
- [ ] 第3章 3.5 节：A0-A3 结果表与 A3 退化机制解释。
- [ ] 第3章 3.6 节：A3R1 结果回填与决策结论。
- [ ] 第1章 1.4 节：根据实证更新“创新点陈述”。

---

## 8. 交付清单（本周末）

1. `work_dir/eval_metrics/A3R1_summary.json`
2. 更新后的 `docs/EXPERIMENT_LOG.md`
3. 第3章初稿（可直接进入整稿）
