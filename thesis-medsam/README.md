# 武汉大学专业硕士学位论文模板（精简版）

## 📁 文件结构

```
thesis-medsam/
├── thesis.tex              # 主文件（已配置为专业硕士）
├── whu-thesis.cls          # 论文文档类
├── module/                 # 模块文件夹
├── logo/                   # 武汉大学校徽等图片
├── data/                   # 参考文献样式文件
├── pages/                  # 论文章节
│   ├── chapter1.tex        # 第一章：绪论
│   ├── chapter2.tex        # 第二章：相关理论与技术
│   ├── chapter3.tex        # 第三章：Balance Loss 方法与实验
│   ├── chapter4.tex        # 第四章：跨病例注意力融合机制
│   ├── chapter5.tex        # 第五章：综合实验设计与结果组织
│   ├── chapter6.tex        # 第六章：总结与展望
│   └── appendix.tex        # 附录
└── ref/                    # 参考文献
    └── references.bib      # 参考文献数据库
```

## 🚀 快速开始

### 1. 修改论文信息

编辑 `thesis.tex` 文件的第 6-30 行，填写您的个人信息：

```latex
\whusetup{
  info = {
    title      = {您的论文题目},
    author     = {您的姓名},
    student-id = {您的学号},
    supervisor  = {导师姓名},
    ...
  }
}
```

**专业学位特别提醒**：如果有校外导师，取消这两行的注释并填写信息：
```latex
supervisor-outer = {校外导师姓名},
academic-title-outer = {校外导师职称},
```

### 2. 编写论文内容

- `pages/abstract.tex` - 中文摘要正文（由模板自动读取）
- `pages/enabstract.tex` - 英文摘要正文（由模板自动读取）
- `pages/chapter1.tex` - 绪论
- `pages/chapter2.tex` - 相关理论与技术
- `pages/chapter3.tex` - Balance Loss 方法与实验
- `pages/chapter4.tex` - LoRA 消融验证与局部特征适配器设计
- `pages/chapter5.tex` - 综合实验与结果分析
- `pages/chapter6.tex` - 总结与展望
- `pages/appendix.tex` - 附录正文（主文件中使用 `\appendix` 后再 `\include`，此文件内不再手动写 `\chapter{附录}`）

### 3. 添加参考文献

在 `ref/references.bib` 中添加参考文献，然后在正文中使用 `\cite{文献标识}` 引用。

### 4. 编译论文

#### 本地编译（需要安装 TeX Live 或 MiKTeX）

```bash
xelatex thesis
bibtex thesis
xelatex thesis
xelatex thesis
```

或使用 latexmk：
```bash
latexmk -xelatex thesis
```

#### Overleaf 在线编译

1. 将整个文件夹打包成 zip
2. 上传到 Overleaf
3. 设置编译器为 **XeLaTeX**
4. 设置主文件为 **thesis.tex**
5. 将 `thesis.tex` 中的中文字体设置改为：
   ```latex
   cjk-font = fandol,
   ```

## ⚙️ 中文字体配置

在 `thesis.tex` 的 `style` 配置中根据平台选择：

- **Windows（当前本地环境默认）**: `cjk-font = windows`
- **Overleaf/Linux**: `cjk-font = fandol`
- **Mac**: `cjk-font = mac`

## 📄 摘要、附录与参考文献说明

- 中文摘要由模板自动读取 `pages/abstract.tex`，英文摘要由模板自动读取 `pages/enabstract.tex`；主文件中无需手动再插入摘要章节。
- 当前主文件采用 `\appendix` 后再 `\include{pages/appendix.tex}` 的方式组织附录，因此 `pages/appendix.tex` 中应直接书写附录内容，不再手动添加 `\chapter{附录}`。
- 当前模板配置使用 `bib-backend = bibtex` 与 `bib-style = numerical`，全文说明与最终输出均以顺序编码制为准。

## 📌 与完整版的区别

本精简版已删除：
- 测试文件（test/）
- 构建脚本（scripts/）
- 参考资料（reference/）
- 用户手册源代码（whudoc.cls, whuthesis.dtx）
- 示例的复杂内容

**保留了所有编译所需的核心文件**，可以直接使用。

## 🔗 相关链接

- 完整模板：https://github.com/whutug/whu-thesis
- Overleaf：https://www.overleaf.com/

---

**祝论文写作顺利！** 🎓
