# Constraint Tax 最终验证报告（2026-06-05 ~ 06-09）

## 1. 问题定义

**Constraint Tax（约束税）**：开放权重模型在面对 `tools` + `response_format`（JSON Schema）双重约束时，工具调用被系统性压制至 0%。模型在训练中学到了"有 Schema 约束 = 直接生成 JSON"的捷径，推理时跳过工具调用，用训练数据记忆或伪造内容填充 Schema。

**核心表现**：

```
tools=[websearch, knowledge_base, ...]
response_format={"type":"json_schema", ...}

GPT-5.4-mini: "先调工具搜资料 → 再填 JSON" ✅
Qwen 等:      "Schema 要 JSON → 直接填（编造数据）" ❌
```

---

## 2. 完整测试矩阵

### 2.1 测试条件

- 测试脚本：`tests/fz-qwen-test/test_*_tool_rfmt.py`
- 每组 5 轮，`stream=true, temperature=0.5`
- 3 组对照：
  - **T1**：`tools=ON, response_format=OFF`（基线：能否调工具）
  - **T2**：`tools=ON, response_format=ON`（关键：双重约束下能否调工具）
  - **T3**：`tools=OFF, response_format=ON`（对照：能否输出合法 JSON）

### 2.2 结果总表

| # | 模型 | 参数量 | 框架/平台 | GPU | T1 工具率 | T2 工具率 | T3 JSON率 | Constraint Tax? |
|---|------|--------|-----------|-----|:---:|:---:|:---:|:---:|
| 1 | **GPT-5.4-mini** | — | OpenAI/AEP 云端 | — | 100% | **100%** | 100% | ✅ **唯一豁免** |
| 2 | Qwen3.6-35B-A3B | 35B MoE | SGLang 0.5.9 | 2×A800 | 100% | **0%** | 100% | ❌ |
| 3 | Qwen3.6-35B-A3B | 35B MoE | vLLM 0.22.0 | 2×A800 | 100% | **0%** | 80% | ❌ |
| 4 | Qwen3.5-122B-A10B | 122B MoE | SGLang 0.5.9 | 2×A800 | 100% | **0%** | 100% | ❌ |
| 5 | GPT-OSS-20B | 20B Dense | SGLang 0.5.9 | 2×A800 | 100% | **0%** | 100% | ❌ |
| 6 | NVIDIA Nemotron 3 Super | 120B MoE | vLLM 0.22.0 | 2×A800 | 100% | **0%** | 100% | ❌ |
| 7 | Qwen3-235B-A22B | 235B MoE | vLLM 0.22.0 | 2×A800 | 0%* | 0%* | 100% | ⚠️ OOM |
| 8 | Qwen3.5-397B-A17B | 397B MoE | 硅基流动云端 | — | 100% | **0%** | 100% | ❌ |
| 9 | Qwen3-VL-235B-A22B-Thinking | 235B MoE | 阿里百炼云端 | — | 100% | **0%** | 100% | ❌ |

> \* 235B 本地部署因显存不足（78.7GB/80GB ×2），`max-model-len=4096`，基线也无法调工具，属于部署条件不足。

### 2.3 已排除因素

| 因素 | 验证方法 | 结论 |
|------|---------|------|
| 推理框架 | SGLang 0.5.9 / vLLM 0.22.0 对比 | ❌ 非根因 |
| 模型规模 | 35B → 122B → 235B → 397B | ❌ 零改善 |
| 模型架构 | Qwen MoE / GPT-OSS Dense / Nemotron Hybrid Mamba-MoE | ❌ 均复现 |
| 量化方式 | FP16 / GPTQ-Int4 / AWQ-Int4 / FP8 | ❌ 均复现 |
| Schema 复杂度 | 1 字段 → 20 字段 | ❌ 无阈值，1 字段即触发 |
| 工具强制 | `tool_choice="required"` / `named` | ❌ 模型完全冻结 |
| Thinking 模式 | qwen3-vl-235b-a22b-thinking | ❌ thinking 不解此问题 |
| 云端全精度 | 硅基流动 / 阿里百炼 | ❌ 云端同样 0% |

### 2.4 行为特征

在 `response_format` 约束下，模型不会报错或拒绝，而是：

- Qwen122B: `"buyer_background": "Simulated Websearch Results"` — 把"该调什么工具"写成了字段值
- Qwen35B(vLLM): 输出"正在分析询盘..."伪进度卡片
- GPT-OSS-20B: `"recommendations": "", "key_findings": []` — 空值填充
- Experiment B: 5/5 回答 `need_search: true`，但 0/5 实际调工具（行为被压制，能力未缺失）

---

## 3. 根因分析

Constraint Tax 是**训练方法论层面的行为固化**，与以下因素均无关：

- ❌ 推理框架（SGLang vs vLLM 结果一致）
- ❌ 模型规模（35B → 397B 零改善）
- ❌ 模型架构（Qwen MoE / GPT-OSS Dense / Nemotron Hybrid 均复现）
- ❌ 量化方式（FP16 / Int4 / FP8 均复现）
- ❌ 部署环境（本地 vs 云端结果一致）

**为什么 GPT-5.4-mini 是例外**：OpenAI 在 RLHF/instruction tuning 阶段针对"有 Schema 时仍优先调工具"做了针对性优化。开放权重模型的训练数据中缺乏这种联合场景的充分覆盖。

---

## 4. 解决方案

### 4.1 方案 A：Decoder-Level Schema Enforcement（XGrammar）— ❌ 不可行

**思路**：不传 `response_format`（避免 Prompt 层触发 Constraint Tax），改用 `guided_json`（XGrammar 解码层约束）。

**验证结果**：
- SGLang 0.5.9 `extra_body.json_schema`: 工具率 100% ✅ 但 JSON 约束不生效 ❌
- vLLM 0.22.0 `extra_body.guided_json`: 工具率 100% ✅ 但 JSON 约束不生效 ❌

**结论**：两个框架的 OpenAI-compatible 端点均不支持"纯 decoder 层约束"模式。方案 A 不可行。

### 4.2 方案 B：框架层透明 Two-Pass — ✅ 已实现并验证

**思路**：将 `response_format` 的生效时机从"第一轮 LLM 调用"推迟到"工具调用完成后"。框架层自动完成，对 Agent 配置和业务逻辑完全透明。

```
传统（触发 Constraint Tax）:
  Step 1: LLM(tools=ON, rfmt=ON) → 被压制 → 直接输出编造 JSON

Plan B:
  Step 1-N: LLM(tools=ON, rfmt=OFF) → 自由调工具 → 执行
  Step N+1: LLM(tools=ON, rfmt=ON)  → 基于真实数据填 JSON
```

**实现**：
- 分支：`AIPRD-317-response-format-json-B1`
- `_InnerAgent._deferred_response_format`：存储待延迟激活的 Schema
- `_MAX_TOOL_STEPS=6`：防止工具阶段无限循环
- `_plan_b_tools_phase`：抑制工具阶段中间文本流到前端
- 自动检测：`is_gpt5_model()==False` 且 `output_format=="blocks"` → 自动启用

**验证结果**：工具调用 5-8 次 + 6 张 Blocks JSON 卡片，端到端通过。

详见：[14-plan-b-design-doc-0608.md](14-plan-b-design-doc-0608.md)

### 4.3 LoRA 微调 — 保底方案

Experiment B 证实模型"大脑"知道该调工具（100% `need_search: true`），只是执行层被压制。SFT + DPO 可针对性地纠正这个执行优先级排序错误。已规划在 [10-GPT-analysis-cuurent-results.md](10-GPT-analysis-cuurent-results.md) 第二阶段。

---

## 5. 模型搜索结论

| 候选 | 2×A800 可部署？ | 突破 Constraint Tax？ | 结论 |
|------|:---:|:---:|------|
| Qwen3.6-35B-A3B | ✅ | ❌ | 当前主力 + Plan B |
| Qwen3.5-122B-A10B | ✅ | ❌ | 规模无改善 |
| Nemotron 3 Super (120B) | ⚠️ 需 INT4 量化 | ❌ | 行为模式不匹配（探索型 vs 服从型） |
| Qwen3-235B-A22B | ❌ 显存不足 | 无法验证 | 上下文 4096 不可用 |
| Qwen3.5-397B-A17B | ☁️ 云端 | ❌ | 397B 也无法突破 |
| **GPT-5.4-mini** | ☁️ 云端 | ✅ | **唯一可用** |

详见：[12-model-search-analysis-0608.md](12-model-search-analysis-0608.md)

---

## 6. 最终结论

1. **Constraint Tax 是开放权重模型的系统性缺陷**。已验证 5 个模型族（Qwen 35B/122B/235B/397B、GPT-OSS-20B、Nemotron 3 Super）、2 个推理框架（SGLang/vLLM）、3 种量化方式（FP16/Int4/FP8）、本地和云端部署——全部复现。

2. **GPT-5.4-mini 是目前唯一已知的同时满足 Tool Calling + Structured Output 的模型**。100% 工具率 + 100% JSON 合法性，跨框架、跨部署方式均验证通过。

3. **Plan B（框架层透明 Two-Pass）是当前最优工程方案**。对 Agent 配置和业务逻辑零改动，已在 B1 分支实现并端到端验证通过。

4. **寻找替代模型的路已经走到尽头**。从 35B 到 397B、从 Qwen 到 Nemotron 到 GPT-OSS，没有任何开放权重模型能够豁免 Constraint Tax。LoRA 微调是唯一可能改变这一局面的保底方向。

---

## 7. 附录：完整测试记录

| 日期 | 测试 | 文档/分支 |
|------|------|----------|
| 06-04 | Qwen35B SGLang 初始测试 | 06-qwen-toolcalling-formatoutput.md |
| 06-05 | 122B / GPT-OSS / vLLM 跨框架验证 | 09-constraint-tax-final-conclusion.md |
| 06-05 | Experiment A/B/C 诊断实验 | 10-GPT-analysis-cuurent-results.md |
| 06-08 | Plan A SGLang + vLLM guided_json 验证 | -A1 分支 |
| 06-08 | Plan B 框架层 Two-Pass 实现 | -B1 分支 |
| 06-08 | Nemotron 3 Super 测试 | 12-model-search-analysis-0608.md |
| 06-09 | Qwen3-235B 本地部署（OOM） | 0609-1.txt |
| 06-09 | 硅基 397B + 阿里 235B-Thinking 云端测试 | 本文档 |
| 06-09 | GPT-5.4-mini AEP 基线再确认 | 本文档 |
