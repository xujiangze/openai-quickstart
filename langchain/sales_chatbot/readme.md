# 人工预处理知识库功能
预先操作, 将问答知识库文件转化为embedding向量化之后的中间文件, 再转化为faiss文件, 以便于后续的快速查询.
```bash
# 将问答知识库文件转化为embedding向量化之后的中间文件
python EmbeddingHelper.py --make_embedding_csv
# 测试加载本地的向量文件,并初始化faiss进行query测试
python FaissHelper.py -- test_query
# 测试将已经保存的向量文件,转化为faiss保存到本地
python FaissHelper.py --test_save --save_dir data/car
# 测试将本地的faiss文件加载到内存,并进行query测试
python FaissHelper.py --test_load
```

# 启动个性化小助理
```bash
python sales_chatbot.py --faiss_dir data/car/faiss
```