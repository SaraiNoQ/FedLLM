import os
from collections import Counter

def count_local_silo_tasks(data_dir="../natural-instructions"):
    # 1. 定义 Silo 映射关键字 (按照优先级排序)
    # SILO_MAPPING = {
    #     "Reasoning_Silo": ["math", "logic", "code", "program", "arithmetic", "reasoning"],
    #     "Translation_Silo": ["translation", "translate", "cross_lingual"],
    #     "Classification_Silo": ["classification", "sentiment", "nli", "topic", "stance"],
    #     "Extraction_Silo": ["extraction", "ner", "tagging", "qa", "question_answering"],
    #     "Generation_Silo": ["generation", "story", "poem", "summarization", "abstractive"],
    #     "Edit_Silo": ["rewriting", "paraphrase", "correction", "gec", "simplification"]
    # }
    SILO_MAPPING = {
        # 1. 问答类 (从 Extraction 拆分，量大且独立)
        "QA_Silo": ["qa", "question_answering", "reading_comprehension", "answer_generation", "context"],
        # 2. 信息抽取类 (NER, 标签等)
        "Extraction_Silo": ["extraction", "ner", "tagging", "identification", "labeling", "slot", "entity"],
        # 3. 情感与评价类 (从 Classification 拆分，具有主观属性)
        "Sentiment_Silo": ["sentiment", "emotion", "opinion", "toxicity", "hate_speech", "irony", "rating"],
        # 4. 逻辑与推理类 (合并 Math, Code, NLI)
        "Logic_Reasoning_Silo": ["math", "logic", "code", "program", "arithmetic", "reasoning", "nli", "entailment", "inference"],
        # 5. 语言转换类 (合并 Translation, Edit, Paraphrase, GEC)
        "Language_Trans_Silo": ["translation", "translate", "rewriting", "paraphrase", "correction", "gec", "simplification", "editing", "style_transfer"],
        # 6. 创作与摘要类 (生成任务)
        "Generation_Silo": ["generation", "summarization", "story", "poem", "write", "creative", "title", "abstractive", "caption"],
        # 7. 基础分类类 (剩余的客观分类任务)
        "Categorization_Silo": ["classification", "classify", "topic", "category", "intent", "detection"]
    }

    if not os.path.exists(data_dir):
        print(f"错误: 找不到目录 {data_dir}")
        return

    # 2. 获取目录下所有符合 NI 命名的任务文件
    # NI 任务文件通常以 task 开头，.json 结尾
    all_files = [f for f in os.listdir(data_dir) if f.startswith("task") and f.endswith(".jsonl")]
    
    # 提取唯一的任务名 (去除 _train.json, _test.json 等后缀以防重复统计同一任务)
    # 例如从 task833_poem_sentiment_classification_train.json 提取 task833_poem_sentiment_classification
    unique_tasks = set()
    for f in all_files:
        # 移除后缀，保留核心任务名
        base_name = f.replace("_train.jsonl", "").replace("_test.jsonl", "").replace("_validation.jsonl", "").replace(".jsonl", "")
        unique_tasks.add(base_name)

    # 3. 统计归类
    silo_counts = Counter()
    unclassified_tasks = []

    for tname in unique_tasks:
        assigned = False
        tname_lower = tname.lower()
        
        for silo, keywords in SILO_MAPPING.items():
            if any(kw in tname_lower for kw in keywords):
                silo_counts[silo] += 1
                assigned = True
                break 
        
        if not assigned:
            silo_counts["General_Silo"] += 1
            unclassified_tasks.append(tname)

    # 4. 打印统计结果
    total_found = len(unique_tasks)
    print("\n" + "="*50)
    print(f"本地数据集统计报告: {data_dir}")
    print(f"总计唯一任务数: {total_found}")
    print("="*50)
    
    for silo, count in sorted(silo_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_found) * 100 if total_found > 0 else 0
        print(f"{silo:<20}: {count:>4} 任务 ({percentage:>5.2f}%)")
    
    print("-" * 50)
    if unclassified_tasks:
        print(f"未分类 (General_Silo) 示例: {unclassified_tasks[:3]}")
    print("="*50)

if __name__ == "__main__":
    # 如果你的主目录路径不同，请在此处修改
    count_local_silo_tasks("./natural-instructions/train")