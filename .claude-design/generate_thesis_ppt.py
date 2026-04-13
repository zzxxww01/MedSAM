# -*- coding: utf-8 -*-
from pathlib import Path
import shutil
from PIL import Image
import win32com.client

ROOT = Path(r"C:/Users/DELL/Desktop/MedSAM")
REF_PPT = ROOT / "汇报.pptx"
OUT_PPT = ROOT / "thesis-medsam" / "MedSAM论文答辩-v5.pptx"
ASSETS = ROOT / ".claude-design" / "ppt-assets"
REF_MEDIA = ROOT / ".claude-design" / "reference-ppt" / "media"

TEMPLATES_TO_DUPLICATE = [2, 7, 8, 7, 9, 7, 2, 7]

if OUT_PPT.exists():
    try:
        OUT_PPT.unlink()
    except PermissionError:
        pass
shutil.copyfile(REF_PPT, OUT_PPT)

app = win32com.client.Dispatch("PowerPoint.Application")
app.Visible = 1
pres = app.Presentations.Open(str(OUT_PPT), WithWindow=False)


def duplicate_before_last(template_idx):
    dup_range = pres.Slides(template_idx).Duplicate()
    dup_slide = dup_range.Item(1)
    dup_slide.MoveTo(pres.Slides.Count - 1)
    return pres.Slides(pres.Slides.Count - 1)


for template in TEMPLATES_TO_DUPLICATE:
    duplicate_before_last(template)


def set_text(slide, shape_idx, text, font_size=None, bold=None):
    shp = slide.Shapes(shape_idx)
    tr = shp.TextFrame.TextRange
    tr.Text = text
    shp.TextFrame.WordWrap = True
    if font_size is not None:
        tr.Font.Size = font_size
    if bold is not None:
        tr.Font.Bold = -1 if bold else 0
    return shp


def replace_picture(slide, shape_idx, image_path, keep_aspect=True):
    shp = slide.Shapes(shape_idx)
    left, top, width, height = shp.Left, shp.Top, shp.Width, shp.Height
    shp.Delete()
    if keep_aspect:
        with Image.open(image_path) as im:
            iw, ih = im.size
        scale = min(width / iw, height / ih)
        new_w = iw * scale
        new_h = ih * scale
        new_left = left + (width - new_w) / 2
        new_top = top + (height - new_h) / 2
    else:
        new_left, new_top, new_w, new_h = left, top, width, height
    return slide.Shapes.AddPicture(str(image_path), False, True, new_left, new_top, new_w, new_h)


def add_picture_fit(slide, image_path, left, top, width, height):
    with Image.open(image_path) as im:
        iw, ih = im.size
    scale = min(width / iw, height / ih)
    new_w = iw * scale
    new_h = ih * scale
    new_left = left + (width - new_w) / 2
    new_top = top + (height - new_h) / 2
    return slide.Shapes.AddPicture(str(image_path), False, True, new_left, new_top, new_w, new_h)


def delete_indices(slide, indices):
    for idx in sorted(indices, reverse=True):
        slide.Shapes(idx).Delete()


def add_caption(slide, left, top, width, height, text, font_size=12):
    box = slide.Shapes.AddTextbox(1, left, top, width, height)
    tr = box.TextFrame.TextRange
    tr.Text = text
    tr.Font.Size = font_size
    tr.ParagraphFormat.Alignment = 2
    return box


def add_textbox(slide, left, top, width, height, text, font_size=13, bold=False):
    box = slide.Shapes.AddTextbox(1, left, top, width, height)
    tr = box.TextFrame.TextRange
    tr.Text = text
    tr.Font.Size = font_size
    tr.Font.Bold = -1 if bold else 0
    box.TextFrame.WordWrap = True
    return box


def replace_bottom_logo(slide, shape_idx):
    replace_picture(slide, shape_idx, REF_MEDIA / "image2.png", keep_aspect=False)


# Slide 1 - Cover
slide = pres.Slides(1)
set_text(slide, 2, "基于平衡损失与多尺度局部适配器的MedSAM腹部多器官分割方法研究", 24, True)
set_text(slide, 3, "周贤玮\r计算机学院 / 电子信息\r指导教师：陈刚 教授\r2026 / 03", 18, False)
replace_bottom_logo(slide, 4)

# Slide 2 - Section
slide = pres.Slides(2)
set_text(slide, 2, "研究背景", 28, True)
replace_bottom_logo(slide, 3)

# Slide 3 - Background
slide = pres.Slides(3)
set_text(slide, 3, "研究背景", 24, True)
set_text(slide, 5, "医学图像分割是计算机辅助诊断、术前规划和放疗勾画的基础能力。MedSAM 将 SAM 迁移到医学场景，但在腹部多器官 CT 中仍存在类别不平衡与边界高频特征不足两类瓶颈。", 15, False)
set_text(slide, 6, "外部因素：扫描条件差异、伪影与噪声会干扰器官边界。\r内部因素：小器官体积小、边界模糊、类别内差异大，进一步加剧学习困难。", 14, False)
delete_indices(slide, [4, 7, 8, 9, 10])
add_picture_fit(slide, ASSETS / "medsam_arch.png", 180, 220, 600, 150)

# Slide 4 - Problem
slide = pres.Slides(4)
set_text(slide, 3, "问题", 24, True)
set_text(slide, 4, "MedSAM 在腹部多器官分割中的三点核心挑战：\r\r1）前景稀疏，背景梯度容易主导训练；\r2）困难像素少，边界与低对比区域学习不足；\r3）ViT 偏全局建模，局部高频边界响应不足。", 15, False)
set_text(slide, 5, "研究设定：使用 GT box prompt，在统一 prompt 条件下评估分割模块本身的优化效果；A0 即原始 MedSAM baseline。", 14, False)
set_text(slide, 6, "技术路线", 16, True)
set_text(slide, 7, "A0 基线 → Balance Loss 消融与修正 → LoRA 负向验证 → MSL-Adapter 结构增强。", 14, False)
replace_picture(slide, 8, ASSETS / "tech_roadmap.png")
slide.Shapes(9).Delete()
add_textbox(slide, 470, 350, 360, 95, "- 训练：FLARE22 3621 + AMOS22 CT 1789\r- 评估：FLARE22 40 个病例\r- 指标：DSC / HD95 / ASD", 13, False)

# Slide 5 - Research overview
slide = pres.Slides(5)
set_text(slide, 3, "研究内容", 24, True)
set_text(slide, 5, "Balance Loss", 16, True)
set_text(slide, 7, "MSL-Adapter", 16, True)
set_text(slide, 9, "核心目标：解决类间/类内双重不平衡。\r\r• Inter-CBL：抑制背景主导\r• Intra-CBL：强化困难像素\r• 两阶段训练：缓解冷启动噪声", 12, False)
set_text(slide, 10, "核心目标：补足 ViT 的局部高频特征。\r\r• 双路径深度可分离卷积\r• 标准卷积 + 膨胀卷积\r• 额外参数仅 0.15%", 12, False)
replace_picture(slide, 6, ASSETS / "balance_loss_flow.png")
replace_picture(slide, 8, ASSETS / "lg_adapter_arch.png")

# Slide 6 - Balance Loss
slide = pres.Slides(6)
set_text(slide, 3, "方法一：Balance Loss", 22, True)
set_text(slide, 5, "", 12, False)
set_text(slide, 6, "Balance Loss 面向 box prompt 二值分割中的双重不平衡：\r\r• Inter-CBL：挖掘困难背景\r• Intra-CBL：强化困难像素\r• Dice：保证全局重叠\r• 两阶段训练：先稳定，再完整平衡\r\r最终配置：α=0.5，τ=0.9。", 13, False)
replace_picture(slide, 7, ASSETS / "balance_loss_flow.png")

# Slide 7 - A3 degradation
slide = pres.Slides(7)
set_text(slide, 3, "A3退化与假设检验", 22, True)
set_text(slide, 4, "退化现象", 16, True)
set_text(slide, 5, "A3 直接组合后明显退化：\r• DSC：0.9526 → 0.9035\r• HD95：3.3684 → 7.9229\r• ASD：0.3749 → 0.8868", 13, False)
set_text(slide, 6, "结论", 16, True)
set_text(slide, 7, "提出两个竞争假设：\r• H1：α 过大\r• H2：切换过早\r\rA3R3 证明主要原因是 α 过大，并得到最优配置。", 13, False)

# Slide 8 - LoRA ablation
slide = pres.Slides(8)
set_text(slide, 3, "LoRA消融与设计动机", 22, True)
set_text(slide, 4, "C2：冻结主干的局限性", 16, True)
set_text(slide, 5, "A3R3 + LoRA(r=4) 且冻结 Image Encoder 后：\r• DSC：0.9596 → 0.8796\r• HD95：2.2511 → 7.7548\r• ASD：0.2463 → 0.9965", 13, False)
set_text(slide, 6, "设计动机", 16, True)
set_text(slide, 7, "复杂多器官任务需要更充分的特征重塑能力。\r因此本文转向“开放主干更新 + 轻量局部适配器”的路线。", 13, False)

# Slide 9 - MSL-Adapter
slide = pres.Slides(9)
set_text(slide, 3, "方法二：MSL-Adapter", 22, True)
replace_picture(slide, 4, ASSETS / "lg_adapter_arch.png")
replace_picture(slide, 5, ASSETS / "adapter_pipeline.png")
add_caption(slide, 65, 488, 320, 22, "模块结构：双路径深度可分离卷积", 11)
add_caption(slide, 435, 488, 320, 22, "插入位置：Encoder 与 Decoder 之间", 11)

# Slide 10 - Section
slide = pres.Slides(10)
set_text(slide, 2, "实验设计", 28, True)
replace_bottom_logo(slide, 3)

# Slide 11 - Dataset and protocol
slide = pres.Slides(11)
set_text(slide, 3, "数据集与评估协议", 22, True)
set_text(slide, 4, "数据与预处理", 16, True)
set_text(slide, 5, "训练数据：\r• FLARE22：3621 张前景切片\r• AMOS22 CT：1789 张前景切片\r• 总计：5410 张\r\r统一预处理：窗化、小目标过滤、前景切片筛选、标准化至 1024×1024。", 13, False)
set_text(slide, 6, "训练与测试口径", 16, True)
set_text(slide, 7, "• 评估统一在 FLARE22 40 个病例上进行\r• Prompt Encoder 冻结，训练 200 epochs\r• AdamW，lr=1e-4，batch size=8\r• 指标：DSC / HD95 / ASD", 13, False)

# Slide 12 - Full ablation
slide = pres.Slides(12)
set_text(slide, 3, "完整消融结果", 22, True)
set_text(slide, 4, "关键方案", 16, True)
set_text(slide, 5, "A0  Baseline      DSC 0.9407\rA2  + Intra-CBL   DSC 0.9526\rA3  直接组合退化  DSC 0.9035\rA3R3 Balance Loss DSC 0.9596\rC2  LoRA 冻结主干 DSC 0.8796\rC3  Final         DSC 0.9620", 13, False)
set_text(slide, 6, "结论", 16, True)
set_text(slide, 7, "• A3R3：HD95 2.2511 / ASD 0.2463\r• C3：HD95 2.0834 / ASD 0.2271\r\rBalance Loss 是主要增益来源；\rMSL-Adapter 在高性能区间继续改善边界质量。", 13, False)

# Slide 13 - Findings
slide = pres.Slides(13)
set_text(slide, 3, "关键结果解读", 22, True)
set_text(slide, 4, "总体发现", 16, True)
set_text(slide, 5, "• A3R3 相比 A0：DSC +2.0%，HD95 -53.4%，ASD -54.2%\r• C3 相比 A3R3：DSC +0.25%，边界指标继续下降\r• C3 相比 A0：DSC +2.1%，HD95 -56.9%，ASD -57.8%", 13, False)
set_text(slide, 6, "统计与器官层面", 16, True)
set_text(slide, 7, "• A0 vs A3R3：p=2.1×10^-6\r• A3R3 vs C3：p=1.32×10^-2\r• 困难器官提升更明显：胰腺、食管、十二指肠、双侧肾上腺", 13, False)

# Slide 14 - Qualitative results
slide = pres.Slides(14)
set_text(slide, 3, "定性可视化结果", 22, True)
replace_picture(slide, 4, ASSETS / "mask_comparison.png")
replace_picture(slide, 5, ASSETS / "boundary_detail.png")
add_caption(slide, 65, 488, 320, 22, "代表性切片：A0 / A3R3 / C2 / C3", 11)
add_caption(slide, 435, 488, 320, 22, "边界放大：C3 对高频细节更敏感", 11)

# Slide 15 - Discussion and limitations
slide = pres.Slides(15)
set_text(slide, 3, "讨论与局限性", 22, True)
set_text(slide, 4, "方法价值", 16, True)
set_text(slide, 5, "• 负向实验同样重要：A3、C1、C2 共同帮助定位有效路径\r• 核心价值在于统一设定下相对原始 MedSAM 的稳定提升\r• 方法收益与设计动机一致，具有较好可解释性", 13, False)
set_text(slide, 6, "当前局限", 16, True)
set_text(slide, 7, "• 依赖 GT box prompt，尚未接入自动检测模块\r• 评估仅基于 40 个病例\r• MSL-Adapter 的增益属于高性能区间内的边际改善\r• 跨模态、跨中心和 3D 扩展尚未验证", 13, False)

# Slide 16 - Section
slide = pres.Slides(16)
set_text(slide, 2, "总结与展望", 28, True)
replace_bottom_logo(slide, 3)

# Slide 17 - Summary
slide = pres.Slides(17)
set_text(slide, 3, "工作总结", 22, True)
set_text(slide, 4, "主要贡献", 16, True)
set_text(slide, 5, "1）提出 Balance Loss，系统缓解类间/类内双重不平衡；\r2）LoRA 消融验证复杂医学域下全参微调的必要性；\r3）提出 MSL-Adapter，以极低额外参数补充局部边界建模能力。", 13, False)
set_text(slide, 6, "未来展望", 16, True)
set_text(slide, 7, "• 从 GT box prompt 过渡到自动提示\r• 在多中心、多模态数据上做外部验证\r• 探索多层适配器与混合微调策略\r• 向三维体积分割扩展", 13, False)

# Slide 18 - Thanks
slide = pres.Slides(18)
set_text(slide, 3, "Thanks", 28, True)
replace_bottom_logo(slide, 4)

pres.Save()
pres.Close()
app.Quit()

print(str(OUT_PPT))
