# 知识库功能

该程序目前是简约版本. 尚未使用文档切片. 意在用最短的例子来演示如何来使用知识库.

实现的功能是: 
- 融合的将csv文件中的问答知识库, 转化为embedding向量, 并使用faiss进行快速查询.
- 根据输入的faiss的功能对数据进行查询, 并返回最相近的问题的答案.


```bash
# export OPENAI_API_KEY
export OPENAI_API_KEY=your_openai_api_key


# 将问答知识库文件转化为embedding向量化之后的中间文件
python EmbeddingHelper.py --make_embedding_csv origin_data/car_sales.csv
# 测试加载本地的向量文件,并初始化faiss进行query测试
python FaissHelper.py -- test_query
# 测试将已经保存的向量文件,转化为faiss保存到本地
python FaissHelper.py --test_save --faiss_dir data/car
# 测试将本地的faiss文件加载到内存,并进行query测试
python FaissHelper.py --test_load  --faiss_dir data/car/faiss
```

# 启动个性化小助理
```bash
python sales_chatbot.py --faiss_dir data/car/faiss
```