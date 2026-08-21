# 评估

评估器衡量的是**片段**质量，而不只是每个窗口的分类准确率。

!!! info "论文证据"

    [论文要点](../research/paper.md)页面明确区分 37 条记录的固定外部测试结果与
    内部 TRL 诊断，并展示对应的变体对比、边界与时间线图。

## 运行评估

评估默认外部数据划分：

~~~bash
python evaluate.py --split external_test
~~~

评估指定预测工作簿：

~~~bash
python evaluate.py --split internal_eval --predictions predictions_internal_eval.xlsx
~~~

所需文件包括预测工作簿与对应标注 CSV：

~~~text
predictions_internal_eval.xlsx
data/annotations/internal_eval_annotations.csv
~~~

## 匹配规则

仅当以下条件全部满足时，预测片段才有资格与参考片段匹配：

1. 两个片段属于同一用户；
2. 两个片段具有相同活动类别；
3. 二者交并比大于 0.5。

匹配是一对一的，因此一个预测不能解释多个参考片段，反之亦然。

对于预测区间 `P` 与参考区间 `G`：

> **IoU(P, G) = duration(P ∩ G) / duration(P ∪ G)**

随后，评估器报告片段级精确率、召回率和 F1：

> **F1 = 2 × precision × recall / (precision + recall)**

项目默认使用用户级片段 F1 的均值作为汇总指标，避免长会话用户自然地主导分数。

## 解释错误

| 失败模式 | 对指标的影响 | 典型诊断 |
| --- | --- | --- |
| 漏检活动 | 假阴性 | 召回率下降 |
| 虚假活动 | 假阳性 | 精确率下降 |
| 类别正确、边界不佳 | IoU 可能不达标 | 精确率和召回率均下降 |
| 预测碎片化 | 产生额外未匹配片段 | 精确率下降 |
| 合并相邻活动 | 一对一匹配冲突 | 召回率与精确率都可能下降 |
| 活动类别错误 | 无可匹配片段 | 精确率和召回率均下降 |

## 评估规范

- 使用 `internal_eval` 校准后处理。
- 将 `external_test` 保留给预定的最终评估流程。
- 记录检查点哈希、归一化资产、集成配置及所有已修改的策略阈值。
- 尽可能使用固定窗口概率比较时序策略，以将解码器变化与模型重训练分离。
- 不要根据网站推断性能。应在经授权的数据划分上运行已发布评估器，并报告准确的
  资产／配置版本。

## 实验输出

可复现封装脚本将评估与诊断材料写入：

~~~text
experiments/results/
experiments/figures/
experiments/logs/
~~~

这些是生成产物，除非明确整理用于发布，否则保留在本地。
