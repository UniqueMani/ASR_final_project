# ASR Final Project — 多语种/方言语音识别四阶段探索

> 本仓库汇总了语音识别课程期末项目的四个子项目：从**方言口音识别**的早期尝试出发，逐步转向**多语言识别**，并通过引入更强的模型与更科学的训练策略不断提升效果，最终在相似语种（东亚语言）识别任务上取得 **80%** 的准确率。

---

## 项目亮点

- **探索路径清晰**：RAMC（口音难、ACC低）→ FLEURS（语言差异大、ACC≈45%）→ 两条细分路线：ECAPA-TDNN（国内语种、ACC≈60–70%）与 Common Voice（相似语种、ACC≈80%）
- **方法逐步增强**：传统 MFCC + GMM/LR 基线 → 深度嵌入（ECAPA-TDNN）→ 预训练表示（wav2vec2）+ 训练策略优化
- **工程化落地**：包含数据处理脚本、训练与评测脚本、可视化与（部分阶段）演示 Demo 代码

---

## 仓库结构

```text
ASR_final_project/
├── ramc_dialect_asr/       # 子项目1：MagicData-RAMC 方言/口音识别 +（可选）实时展示原型
├── fleurs_langid/          # 子项目2：FLEURS 多语言识别（经典方法：MFCC + GMM/LR 等）
├── ECAPA-TDNN/             # 子项目3：国内语种识别（ECAPA-TDNN 嵌入 + 分类器）
└── CommonVoice24.0_asr/    # 子项目4：相似语种识别（CNN/BiLSTM/wav2vec2 等）
```

> 各子项目目录内通常包含：数据准备/清单生成脚本、训练脚本、评测脚本、输出结果目录（log/figures/results 等）。


## 快速开始

> 各子项目的环境和命令可能略有差别，请以子项目目录内脚本/README 为准。

### 1) 创建环境（示例）
```bash
conda create -n asr_proj python=3.10 -y
conda activate asr_proj
pip install -r requirements.txt   # 若各子项目分别有 requirements，请进入对应目录安装
```

### 2) 数据准备
- **RAMC**：从 OpenSLR 下载 MagicData-RAMC，并按 `ramc_dialect_asr/` 的脚本说明组织目录  
- **FLEURS**：使用脚本下载/合并指定语言子集（见 `fleurs_langid/scripts/`）  
- **ECAPA-TDNN**：准备国内语种数据（普通话/藏语/维语等），统一采样率与时长；按脚本生成 manifest  
- **Common Voice**：下载对应版本（如 v24.0）指定语言数据；用脚本完成采样、划分与特征/缓存生成  

> 注意：请确保训练/测试集按 **speaker 独立** 划分（同一说话人不要同时出现在训练与测试），否则会高估模型效果。

### 3) 训练与评测
```bash
# 训练
python scripts/train.py --config configs/xxx.yaml

# 评测
python scripts/eval.py --ckpt outputs/best.ckpt
```

## 方法改进总结

- **任务难度控制**：从“口音差异细微”转向“语种差异更明显”的任务，降低不可控因素  
- **表示能力升级**：从 MFCC 这类手工特征 → 深度嵌入（ECAPA）→ 预训练表征（wav2vec2）  
- **训练策略逐步规范**：数据均衡采样、speaker‑independent 划分、早停/学习率策略、（可选）数据增强、固定随机种子复现实验  
- **误差分析驱动迭代**：重点关注“相似语种”混淆对（如 普通话 vs 粤语，日语 vs 韩语），针对性改进模型与训练

---

## 许可与致谢

- 数据集：MagicData‑RAMC（OpenSLR）、FLEURS（Google）、Mozilla Common Voice 等
- 预训练模型：wav2vec2 / ECAPA‑TDNN 等

---

## 联系方式 / 复现实验说明

如需复现，请优先参考各子项目目录内脚本参数与readme，并在 issue 中附上：
- OS / Python 版本
- 依赖版本（pip freeze）
- 运行命令与报错日志（尽量完整）
