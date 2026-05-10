# CCFL项目记录 - IJCNN论文追踪
# 创建时间: 2026-05-10
# 最后更新: 2026-05-10
# 目标: 诚实评估创新性，目标IJCNN录用率70-80%

## ============================================================
## 一、项目核心方法
## ============================================================

### 方法名称: CCFL (Context-Conditional Feature Learning)
### 核心思想:
#   训练编码器 f(x,c) = g(h(x) || e(c))，使得同一视觉输入在不同上下文中
#   产生不同的特征表示。通过监督对比学习训练，使用(context_id * 10 + digit)
#   作为标签，同时保留上下文区分和内容区分。
#   测试时通过多假设解码(Multi-Hypothesis Decoding)推断上下文，无需上下文ID。
#   顺序学习时使用Replay Buffer防止遗忘。

### 神经科学动机:
#   MEC(内嗅皮层)提供上下文信号 → DG(齿状回)进行模式分离
#   不同环境/上下文中，相同刺激应产生不同的记忆表征
#   参考: Pilly et al. 2018, "Modeling Contextual Modulation of Memory Associations
#         in the Hippocampus" (Frontiers in Human Neuroscience)

## ============================================================
## 二、实验结果汇总 (全部真实，严禁造假)
## ============================================================

### 实验1: Context-Dependent MNIST (核心实验, 联合训练)
#   场景: 相同数字(0-4)在5个上下文中有不同标签
#   结果 (noise=0):
#     BL-kNN:   19.9%  (无上下文信息的基线)
#     BL-Cond:  99.2%  (已知上下文ID的条件kNN, 上界)
#     CCFL-T:   99.1%  (CCFL+已知上下文ID)
#     CCFL-I:   99.1%  (CCFL+推断上下文, 我们的方法!)
#   关键: CCFL-I ≈ BL-Cond，不需要上下文ID就能达到需要ID的性能

### 实验2: Context-Dependent MNIST (顺序学习+Replay)
#   noise=0.0: CCFL-T=99.2%, CCFL-I=99.0%, CtxInf=99.8%
#   noise=0.2: CCFL-T=99.1%, CCFL-I=93.3%, CtxInf=94.0%
#   noise=0.5: CCFL-T=78.2%, CCFL-I=48.2%, CtxInf=52.5%
#   关键: 顺序学习也能达到接近联合训练的性能!

### 实验3: Split-MNIST 顺序学习 (公平对比)
#   FineTune:    19.8%
#   EWC:         19.8%
#   ER:          87.6%
#   CCFL-Replay: 96.2%  ← 最好!
#   关键: CCFL-Replay >> ER >> EWC ≈ FineTune

### 实验4: Permuted-MNIST 顺序学习 (公平对比)
#   EWC:         95.0%
#   CCFL-Replay: 89.6%
#   注意: CCFL-Replay < EWC，EWC在Permuted-MNIST上更强
#   原因: Permuted-MNIST不需要上下文区分，EWC的参数正则化更直接

### 实验5: 理论界验证
#   delta(上下文间距离) = 1.5722
#   epsilon(上下文内扩散) = 1.0891
#   分离比 delta/epsilon = 1.4436
#   界: P(correct) >= 1 - C*exp(-n*(delta-epsilon)^2/2)
#   n=40时: 经验=1.0, 理论界=0.953, gap=0.047

### 实验6: 新颖性检测
#   Max-sim AUROC: 0.48 (接近随机)
#   结论: 仅靠视觉特征无法检测未知上下文

### 消融: Context Embedding维度
#   ctx_dim=8-64: 都能达到98.4-98.8%，16维已足够

## ============================================================
## 三、网络调研对比分析
## ============================================================

### IJCNN录用率: 35%-42% (2021-2025)
#   2025: 5526投稿, 2152录用, 录用率~39%
#   审稿标准: 技术质量(C1) + 创新性(C2) + 实验充分性(C3) + 表达质量(C4)
#   偏好: 脉冲神经网络(18.3%), 神经形态系统(15.7%), 图神经网络(12.9%)
#   关键: IJCNN更青睐有生物神经科学基础的创新研究!

### 相关工作对比:
#
# 1. GRID (2025): Task-Agnostic Prompt-Based CL
#    - 自动任务识别 + 约束解码，与CCFL的MHD思路类似
#    差异: CCFL是特征空间方法，GRID是prompt空间方法
#
# 2. CoCoOp (CVPR 2022): Conditional Prompt Learning
#    - 输入条件化的prompt，与CCFL的条件编码思路相似
#    风险: 审稿人可能认为CCFL只是CoCoOp在CL领域的应用
#    差异: CoCoOp用于VLM适配，CCFL用于持续学习+记忆检索+task-agnostic推理
#
# 3. Conditional Finetuning (ICML 2024): 条件语言建模
#    差异: 面向NLP领域，CCFL面向视觉记忆检索
#
# 4. HippoRAG 2 (ICML 2025): 海马体启发的RAG
#    差异: HippoRAG是检索增强，CCFL是特征学习
#
# 5. Pilly et al. 2018: 海马体上下文调制模型
#    - CCFL的直接神经科学前驱
#    差异: Pilly是计算模型，CCFL是深度学习实现+实验验证
#
# 6. Dual-Memory CL (TMLR 2025): 短期+长期记忆框架
#    差异: 关注记忆分配，CCFL关注特征表示
#
# 7. Saighi & Rozenberg 2025: 联想记忆网络中的自主检索
#    差异: 抑制性可塑性，CCFL是对比学习

## ============================================================
## 四、诚实创新性评估 (更新后)
## ============================================================

### 真正的创新点:
# 1. [强] Context-Conditional Feature Learning用于task-agnostic CL
#    - 同一输入+不同上下文→不同特征，无需test-time上下文ID
#    - 顺序学习+Replay也能工作(96.2% on Split-MNIST)
#    - 这在CL领域是新的
# 2. [中] Multi-Hypothesis Decoding用于上下文推理
#    - 简单但有效，O(C)复杂度
# 3. [中] 顺序学习CCFL + Replay Buffer
#    - 不冻结backbone，用replay防止遗忘
#    - 在Split-MNIST上96.2% >> ER 87.6%
# 4. [弱] 神经科学动机(MEC-DG)
#    - Pilly 2018已有计算模型

### 仍存在的问题:
# 1. [中等] Permuted-MNIST上CCFL-Replay(89.6%) < EWC(95.0%)
#    - 需要解释为什么在某些场景下不如EWC
# 2. [中等] 仅在MNIST上验证，缺少CIFAR-100
# 3. [中等] 高噪声时性能急剧下降
# 4. [弱] 与CoCoOp的区分需要更清晰

### 与IJCNN录用标准的评估:
# - 技术质量(C1): 顺序学习已修复，Split-MNIST 96.2% → 4/5
# - 创新性(C2): CCFL+MHD+Replay有一定新意 → 3.5/5
# - 实验充分性(C3): MNIST三个变体，但缺CIFAR-100 → 3/5
# - 表达质量(C4): 待写 → N/A
# 综合估计: 当前录用概率 ~50-55%

## ============================================================
## 五、提升到70-80%的关键路径
## ============================================================

### 必须完成:
# 1. 添加CIFAR-100实验 (提升C3到4/5)
# 2. 与更多CL基线对比 (DER++, SCR, L2P等)
# 3. 清晰区分与CoCoOp (强调task-agnostic + 记忆检索)
# 4. 更紧的理论界 (提升C1到4.5/5)

### 可选增强:
# 5. 环境信号辅助的新颖性检测
# 6. 跨上下文知识迁移
# 7. 上下文数量可扩展性实验

## ============================================================
## 六、更新日志
## ============================================================

# 2026-05-10 v1: 创建项目记录
#   - 完成CCFL v2实验(修复表示坍塌)
#   - 完成综合实验(4个数据集)
#   - 完成网络调研(7篇相关论文)
#   - 诚实评估: 当前录用概率25-30%

# 2026-05-10 v2: 修复顺序学习
#   - 使用Replay Buffer替代冻结backbone
#   - Split-MNIST: CCFL-Replay 96.2% >> ER 87.6% >> EWC 19.8%
#   - Permuted-MNIST: CCFL-Replay 89.6% < EWC 95.0%
#   - Context-Dependent Sequential: CCFL-I=99.0% (noise=0)
#   - 诚实评估: 当前录用概率50-55%
#   - 关键瓶颈: 缺CIFAR-100 + Permuted-MNIST不如EWC + 与CoCoOp区分
