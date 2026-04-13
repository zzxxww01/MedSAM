# 组会PPT摘要

总页数：24

## 第1页
- 武汉大学国家网络安全学院
- Segment anything in medical images(MedSAM)
- 周贤玮
- 2026/1/9

## 第2页
- O
- utline
- 01
- Background
- 02
- Method
- 03
- Experiment
- 04
- Conclusion

## 第3页
- Background
- 01

## 第4页
- Background
- 现状：医学分割主要依赖
- U-Net
- 及其变体（如
- nnU-Net
- ）。
- 碎片化：一个器官一个模型，无法泛化。
- 数据昂贵：罕见病灶缺乏标注，难以训练高性能模型。
- 临床部署难：医院需要维护数十个针对不同部位的独立模型。

## 第5页
- Related
- Work
- SAM (Segment Anything Model):
- 自然图像领域的通用分割模型，零样本能力强。
- 失败原因
- (Why not SAM):
- 域差异
- :
- 医学图像（灰度、低对比度）与自然图像（
- RGB
- 、强边缘）特征不同。
- 边界模糊
- :
- 肿瘤往往与周围组织边界不清，
- SAM
- 容易分割错误
- 或
- “
- 欠分割
- ”
- 。

## 第6页
- MedSAM
- 目标
- :
- 构建一个医学图像分割的基础模型。
- 策略
- :
- 将
- SAM
- 从自然图像域迁移到医学域。
- SAM Architecture + Medical Data = MedSAM
- 核心贡献
- :
- 构建了超大规模医学数据集。
- 通过全量微调
- (Full Fine-tuning)
- 使模型适配医学特征。

## 第7页
- Method
- 02

## 第8页
- Dataset
- 数据规模：
- 1,570,263
- 对图像
- -
- 掩码
- (Image-Mask Pairs)
- 。
- 覆盖范围：
- 10
- 种成像模态
- (CT, MRI, X-Ray, US, Pathology
- 等
- )
- 。
- 30+
- 种癌症类型。
- 来源：
- TCIA, MICCAI Challenges, Kaggle
- 等公开数据集的整合。

## 第9页
- Dataset
- 挑战：不同模态的像素值物理意义不同（如
- CT
- 是
- HU
- 值，
- MRI
- 是相对强度）。
- 标准化流程：
- CT
- ：应用特定的窗宽窗位（软组织、肺、脑窗）
- ->
- 归一化。
- MRI/
- 其他：
- 0.5-99.5
- 百分位截断
- ->
- 归一化。
- 尺寸：统一
- Resizing
- 到
- 1024
- ×
- 1024
- ×
- 3
- 。
- 3D
- 处理策略：

## 第10页
- Network Architecture
- 核心组件：
- Image Encoder (
- 图像编码器
- )
- ：基于
- ViT
- ，提取图像特征。
- Prompt Encoder (
- 提示编码器
- )
- ：处理
- Bounding Box
- 提示。
- Mask Decoder (
- 掩码解码器
- )
- ：融合特征并生成分割掩码。
- 数据流：

## 第11页
- Image Encoder Details

## 第12页
- Prompt Engineering
- 为什么选择
- Bounding Box
- ？
- 点提示
- (Point)
- 在医学图像中存在歧义（如血管、神经）。
- Box
- 提供了明确的空间范围约束。
- 训练时的提示模拟：
- 根据
- Ground Truth
- 生成
- Bounding Box
- 。
- 引入
- 0-20
- 像素的随机扰动
- (Jitter)
- ，模拟人工框选的误差。
- Prompt Encoder (Frozen)：
- 输入：
- 机制：位置编码 (Positional Encoding)。将框的角点坐标映射为高维向量。
- 状态：冻结不训练。因为几何位置信息是跨领域通用的。

## 第13页
- Mask Decoder
- Mask Decoder (Trainable)：
- 机制：双向交叉注意力。
- Token-to-Image Attention (提示去‘关注’图像)。
- Image-to-Token Attention (图像去‘修正’提示)。
- 输出：经转置卷积上采样回 1024
- x 1024。

## 第14页
- Loss Function
- Loss Function
- ：
- Dice Loss
- ：关注区域重叠，解决正负样本不平衡。
- CE Loss (Cross-Entropy)
- ：关注像素级分类精度。
- 训练细节：
- Load Checkpoint: 加载 sam_vit_b_01ec64.pth (SA-1B Pre-trained)。
- Freeze: Prompt Encoder。
- Update: Image Encoder + Mask Decoder。
- 20
- ×
- NVIDIA A100 GPUs
- 。
- 150 Epochs
- 。
- Optimizer: AdamW
- 。

## 第15页
- Experiment
- 03

## 第16页
- Validation Set
- 目的：区分
- “
- 拟合能力
- ”
- 与
- “
- 泛化能力
- ”
- 。
- 内部验证
- (Internal Validation)
- ：
- 86
- 个任务，数据分布在训练集中见过
- (Seen Distribution)
- 。
- 外部验证
- (External Validation)
- ：
- 60
- 个全新任务，数据分布完全未见
- (Unseen/Zero-shot)
- 。

## 第17页
- Internal Validation
- 核心结论：
- MedSAM
- 大幅优于
- SAM
- ，且与
- Specialist nnU-Net
- 性能相当；模型在
- 86
- 个任务中的相对排名频率获得
- Rank 1
- 的次数最多。
- 数据：
- MedSAM
- 在
- 86
- 个任务上的
- Median DSC
- 极高。
- MedSAM
- 在大多数任务中都拿到了第一名。

## 第18页
- Internal Validation
- 无论是 CT 中的肝癌，还是内窥镜下的息肉，MedSAM（黄色掩码）都与专家标注（紫红色轮廓）高度重合，而原始 SAM 经常会分割出多余的背景区域。

## 第19页
- External
- Validation
- 现象：
- Specialist nnU-Net
- 性能经常出现大幅下降（
- Distribution Shift
- ）。
- 结论：
- MedSAM
- 保持了极高的稳定性。

## 第20页
- External
- Validation
- 如第二行的宫颈癌
- MRI
- ，这种图像软组织边界非常模糊，不仅
- SAM
- 失败了，连专用的
- DeepLabV3+
- 也分割得很差，但
- MedSAM
- 依然能准确捕捉到肿瘤边界。

## 第21页
- Scaling Law
- 实验：对比
- 10k, 100k, 1M
- 训练数据量的性能。
- 趋势：
- DSC
- 随数据量对数线性增长。
- 结论：医学分割遵循
- Scaling Law
- ，数据越多，模型越强。

## 第22页
- Conclusion
- 04

## 第23页
- Conclusion
- 对比实验：全手动分割
- vs. MedSAM
- 辅助分割。
- 任务：
- 3D
- 肾上腺肿瘤标注。
- 结果：
- MedSAM
- 辅助模式将标注时间减少了
- 82%
- 以上。

## 第24页
- 武汉大学国家网络安全学院
- 敬请批评指正！

