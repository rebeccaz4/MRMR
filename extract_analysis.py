import json

# 你的 JSON 文件路径
json_path = "/home/siyue/Projects/results/KnowledgeAny2AnyRetrieval_default_predictions.json"

# 目标 query id
target_query_id = "query-test-validation_Art_6"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

if target_query_id in data:
    # 取出该 query 对应的完整结果字典
    result_dict = data[target_query_id]
    print("完整结果：")
    for k, v in result_dict.items():
        print(f"{k}: {v}")

    # ---- 进一步处理：按分数从大到小排序并取前 5 ----
    top5 = dict(
        sorted(result_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    )
    print("\nTop 5 结果：")
    for k, v in top5.items():
        print(f"{k}: {v}")

else:
    print(f"未找到 query-id: {target_query_id}")
