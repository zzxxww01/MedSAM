# 附录：历史口径与归档记录（非当前执行主线）

> 目的：保留历史信息，避免污染当前主线文档。  
> 说明：本文件内容用于追溯，不作为当前实验直接执行依据。

---

## 1. 历史执行口径

### 1.1 AMOS22 模板命令（历史）
- 历史规划中存在以 `data/npy/CT_AMOS` 为主的数据流程与命令模板。
- 当前主线已经切换到 `FLARE22 -> data/npy/CT_Abd`。
- 因此 AMOS 模板仅作为“可迁移参考”，不作为当前任务默认入口。

### 1.2 旧阶段时间表（历史）
- 早期文档中包含“1月-5月”整体时间计划。
- 当前执行应以 `docs/WEEKLY_TASKS.md` 与 `docs/FULL_EXPERIMENT_PLAN.md` 为准。

---

## 2. 历史问题与修复记录

### 2.1 A3 初次失败（OOM）
- 现象：`batch_size=2` 在原始 A3 配置下显存不足。
- 修复：改为 `batch_size=1` 后训练可稳定完成。

### 2.2 DDP 环境变量缺失
- 现象：`MASTER_ADDR expected, but not set`，并出现误用 4 卡。
- 修复：统一采用 `nohup bash -lc` + `export CUDA_VISIBLE_DEVICES/MASTER_ADDR/MASTER_PORT` 模板。

### 2.3 命令行换行误触发
- 现象：把参数行单独回车执行，导致 `--exp_name：未找到命令`。
- 修复：一条完整命令一次粘贴，或使用文档中给出的多行模板完整执行。

---

## 3. 历史文档迁移说明

以下信息已迁移或重定位：
1. 论文总控：迁移到 `docs/THESIS_MASTER_GUIDE.md`
2. 章节蓝图：迁移到 `docs/THESIS_PLAN.md`
3. 当前命令：迁移到 `docs/SERVER_COMMANDS.md`
4. 当前路线：迁移到 `docs/FULL_EXPERIMENT_PLAN.md`

---

## 4. 历史内容使用规则

1. 若内容与主线冲突，以主线文档为准。
2. 历史命令需先核对数据路径、任务名、GPU/端口再执行。
3. 历史结果可用于“过程说明”，不可直接作为“最终结论证据”。
