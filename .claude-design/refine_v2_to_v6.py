# -*- coding: utf-8 -*-
from pathlib import Path
import shutil
from PIL import Image
import win32com.client

ROOT = Path(r"C:/Users/DELL/Desktop/MedSAM")
SRC = ROOT / "thesis-medsam" / "MedSAM论文答辩-v2.pptx"
OUT = ROOT / "thesis-medsam" / "MedSAM论文答辩-v6.pptx"
BADGE = ROOT / ".claude-design" / "reference-ppt" / "media" / "image2.png"

if OUT.exists():
    try:
        OUT.unlink()
    except Exception:
        pass

shutil.copyfile(SRC, OUT)

app = win32com.client.Dispatch("PowerPoint.Application")
app.Visible = 1
pres = app.Presentations.Open(str(OUT), WithWindow=False)


def replace_badge_on_slide(slide):
    targets = []
    for i in range(1, slide.Shapes.Count + 1):
        shp = slide.Shapes(i)
        if shp.Type == 13 and shp.Left > 650 and shp.Top > 430 and shp.Width > 220 and shp.Height < 120:
            targets.append((i, shp.Left, shp.Top, shp.Width, shp.Height))
    for i, left, top, w, h in reversed(targets):
        slide.Shapes(i).Delete()
        slide.Shapes.AddPicture(str(BADGE), False, True, left, top, w, h)


# 1) 统一替换模板角标（右下/右上条带）
for sidx in range(1, pres.Slides.Count + 1):
    replace_badge_on_slide(pres.Slides(sidx))

# 2) 融合组会PPT背景内容（重点更新背景页）
# Slide 3: 研究背景
s3 = pres.Slides(3)
s3.Shapes(2).TextFrame.TextRange.Text = "研究背景"
s3.Shapes(3).TextFrame.TextRange.Text = (
    "医学图像分割是辅助诊断、术前规划和放疗勾画的核心基础任务。"
    "传统方法往往一类器官对应一个模型，存在数据依赖强、泛化弱、临床维护成本高的问题。"
)
s3.Shapes(4).TextFrame.TextRange.Text = (
    "SAM 在自然图像中表现强，但直接迁移到医学场景会受模态差异与边界模糊影响。"
    "MedSAM 的目标就是把 SAM 从自然图像迁移到医学图像，并通过全量微调提升可用性。"
)

# Slide 4: 问题与设定（进一步压缩）
s4 = pres.Slides(4)
s4.Shapes(2).TextFrame.TextRange.Text = "问题"
s4.Shapes(3).TextFrame.TextRange.Text = (
    "核心挑战：\r"
    "1）前景像素稀疏，背景梯度易主导；\r"
    "2）困难像素少，边界学习不足；\r"
    "3）ViT 偏全局建模，局部高频细节表达不足。"
)
s4.Shapes(4).TextFrame.TextRange.Text = (
    "本文采用 GT box prompt，在统一提示条件下评估分割模块改进效果；"
    "A0 即原始 MedSAM baseline。"
)

# 3) 再压缩几页文案（避免过满）
s7 = pres.Slides(7)
s7.Shapes(4).TextFrame.TextRange.Text = "A3 直接组合后退化：DSC 0.9526→0.9035；HD95 3.3684→7.9229；ASD 0.3749→0.8868。"
s7.Shapes(6).TextFrame.TextRange.Text = "竞争假设：H1(α过大)、H2(切换过早)。A3R3 结果表明主要原因是 α 过大。"

s8 = pres.Slides(8)
s8.Shapes(4).TextFrame.TextRange.Text = "C2(LoRA+r=4+冻结主干)显著退化：DSC 0.9596→0.8796，说明复杂多器官任务仍需充分特征重塑。"
s8.Shapes(6).TextFrame.TextRange.Text = "因此转向“开放主干更新 + 轻量局部适配器(MSL-Adapter)”路线。"

s12 = pres.Slides(12)
s12.Shapes(4).TextFrame.TextRange.Text = "A0:0.9407  A2:0.9526  A3:0.9035\rA3R3:0.9596  C2:0.8796  C3:0.9620"
s12.Shapes(6).TextFrame.TextRange.Text = "结论：Balance Loss 是主要增益来源；MSL-Adapter 在高性能区间继续改善边界指标。"

pres.Save()
pres.Close()
app.Quit()

print(str(OUT))
