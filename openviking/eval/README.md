# OpenViking Eval 模块

OpenViking 的评估模块，提供 RAG 系统的多维度评估能力。

## 模块作用

Eval 模块支持对 RAG 系统进行全面评估：

- **检索质量评估**：精确度、召回率、相关性
- **生成质量评估**：忠实度、答案相关性
- **性能评估**：检索速度、端到端延迟
- **框架集成**：支持 RAGAS、TruLens 等主流评测工具

## 模块设计

```
openviking/eval/
├── types.py         # 数据类型：EvalSample, EvalDataset, EvalResult
├── base.py          # 评估器基类：BaseEvaluator
├── ragas.py         # RAGAS 框架集成
├── generator.py     # 数据集生成器
└── pipeline.py      # RAG 查询流水线
```

### 核心类型

```python
# 评估样本
EvalSample(
    query="问题",
    context=["检索上下文"],
    response="生成答案",
    ground_truth="标准答案"
)

# 评估数据集
EvalDataset(name="dataset", samples=[...])

# 评估结果
EvalResult(sample=..., scores={"faithfulness": 0.85})
```

### 评估器接口

```python
class BaseEvaluator(ABC):
    async def evaluate_sample(self, sample: EvalSample) -> EvalResult
    async def evaluate_dataset(self, dataset: EvalDataset) -> SummaryResult
```

## 安装方法

```bash
# 基础安装
pip install openviking

# RAGAS 评估支持
pip install ragas datasets
```

## 用法示例

### 示例 1：RAGAS 评估

```python
import asyncio
from openviking.eval import EvalSample, EvalDataset, RagasEvaluator

async def main():
    # 准备评估数据
    samples = [
        EvalSample(
            query="OpenViking 是什么？",
            context=["OpenViking 是上下文数据库..."],
            response="OpenViking 是 AI Agent 数据库",
            ground_truth="OpenViking 是开源上下文数据库"
        ),
    ]
    dataset = EvalDataset(name="eval", samples=samples)
    
    # 运行评估
    evaluator = RagasEvaluator()
    summary = await evaluator.evaluate_dataset(dataset)
    
    # 输出结果
    for metric, score in summary.mean_scores.items():
        print(f"{metric}: {score:.2f}")

asyncio.run(main())
```

### 示例 2：CLI 工具评估

```bash
cd examples/eval

# 基础评估
uv run rag_eval.py \
    --docs_dir ./docs \
    --question_file ./questions.jsonl \
    --output ./results.json

# 启用 RAGAS 指标
uv run rag_eval.py \
    --docs_dir ./docs \
    --question_file ./questions.jsonl \
    --ragas \
    --output ./results.json
```

## 评估指标

| 类别 | 指标 | 说明 |
|------|------|------|
| 检索质量 | context_precision | 上下文精确度 |
| | context_recall | 上下文召回率 |
| 生成质量 | faithfulness | 答案忠实度 |
| | answer_relevance | 答案相关性 |
| 性能指标 | retrieval_time | 检索耗时 |
| | total_latency | 端到端延迟 |

## 相关文件

- CLI 工具：[examples/eval/rag_eval.py](../../examples/eval/rag_eval.py)
- 示例数据：[local_doc_example_glm5.jsonl](./local_doc_example_glm5.jsonl)
- 测试文件：[tests/eval/](../../tests/eval/)
