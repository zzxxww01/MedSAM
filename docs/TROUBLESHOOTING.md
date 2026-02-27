# 服务器端快速操作指南

## 问题：找不到 setup_amos22.sh

### 解决方案1：拉取最新代码

```bash
cd ~/chengang/zxw/MedSAM
git pull
ls -la setup_amos22.sh  # 检查文件是否存在
```

### 解决方案2：直接手动解压（推荐）

如果 git pull 后仍然找不到文件，直接手动解压即可：

```bash
cd ~/chengang/zxw/MedSAM/data
unzip amos22.zip
```

解压后检查目录结构：
```bash
ls -la
```

你会看到类似这样的目录：
- `AMOS22/` 或 `amos22/` 或 `amos/`

记住这个目录名，然后直接运行预处理：

```bash
cd ~/chengang/zxw/MedSAM
python pre_AMOS22.py
```

预处理脚本会自动检测目录名，无需手动修改。

---

## 完整操作流程（从当前位置开始）

```bash
# 当前在 data 目录
unzip amos22.zip

# 查看解压结果
ls -la

# 返回项目根目录
cd ..

# 运行预处理
python pre_AMOS22.py
```
