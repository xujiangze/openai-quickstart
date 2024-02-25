import os
from typing import List, Callable, Iterable, Tuple
from openai import OpenAI
import pandas as pd
import argparse
import tiktoken
import os
import ast
from langchain_community.embeddings import OpenAIEmbeddings

# 本文介绍了, 如何将知识库的csv文件, 转换为embedding向量.

"""
数据包导入说明
1. os: Python 标准库，提供了丰富的方法用于处理文件和目录。
2. pandas: 一个用于数据处理和分析的 Python 库，提供了 DataFrame 数据结构，方便进行数据的读取、处理、分析等操作。
3. tiktoken: OpenAI 开发的一个库，用于从模型生成的文本中计算 token 数量。
"""


class OpenAIEmbedding(object):
    """
    OpenAIEmbedding
    """
    TickTokenEncoding = "cl100k_base"

    def __init__(self, openai_api_key: str, embedding_model: str = "text-embedding-ada-002"):
        self.api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model

    def embedding_text(self, text):
        res = self.client.embeddings.create(input=text, model=self.embedding_model)
        return res.data[0].embedding


class EmbeddingHelper(object):
    """
    EmbeddingHelper
    """
    DataFrameColumnCombined = "__system_combined"
    DataFrameColumnNToken = "__n_token"
    DataFrameColumnEmbedding = "__embedding"

    def __init__(self):
        self.df: pd.DataFrame = pd.DataFrame()

    def read_already_embedding_file(self, file_path: str):
        self.df = self._read_csv(file_path)
        # 将字符串转换为向量, csv 文件读取的内容默认不是向量.是字符串.
        self.df[self.DataFrameColumnEmbedding] = self.df[self.DataFrameColumnEmbedding].apply(ast.literal_eval)

    def embedding_by_origin_file(self, origin_csv_path: str, embedding_encoding: str,
                                 embedding_function: Callable[[str], List[float]]):
        """
        通过原始csv文件进行embedding
        :param origin_csv_path:
        :param embedding_encoding:
        :param embedding_function:
        :return:
        """
        self.df = self._read_csv(origin_csv_path)
        self.df = self._make_combined_column(self.df)
        self._embedding(embedding_encoding, self.df, embedding_function)

    def output(self, output_csv_path: str):
        """
        将embedding结果输出csv文件
        :param output_csv_path:
        :return:
        """
        self.df.to_csv(output_csv_path)

    def get_text_embedding_pairs(self) -> (Iterable[Tuple[str, List[float]]], OpenAIEmbeddings):
        """
        获取文本和embedding的对, 以便进行向量数据库的构建
        :return:
        """
        texts = self.df[EmbeddingHelper.DataFrameColumnCombined].tolist()
        text_embeddings = self.df[EmbeddingHelper.DataFrameColumnEmbedding].tolist()
        text_embedding_pairs = zip(texts, text_embeddings)
        embedding = OpenAIEmbeddings()
        return text_embedding_pairs, embedding

    @staticmethod
    def _read_csv(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, index_col=0)

    @staticmethod
    def _get_combined_row(key: str, value: str):
        return f'{key.strip()}: {value.strip()} \n'

    def _make_combined_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将所有列的值合并为一列
        :param df:
        :return:
        """
        tmp_df = df.copy()
        for index, row in df.iterrows():
            combined_str = ""
            for key, value in row.items():
                if key == self.DataFrameColumnCombined:
                    continue
                combined_str += self._get_combined_row(str(key), value)
            tmp_df.at[index, self.DataFrameColumnCombined] = combined_str

        return tmp_df

    def _embedding(self, embedding_encoding: str, df: pd.DataFrame, embedding_function: Callable[[str], List[float]]) \
            -> pd.DataFrame:
        """
        # 对所有行进行embedding
        # 模型类型
        # 建议使用官方推荐的第二代嵌入模型：text-embedding-ada-002
        embedding_model = "text-embedding-ada-002"
        # text-embedding-ada-002 模型对应的分词器（TOKENIZER）
        embedding_encoding = "cl100k_base"
        # text-embedding-ada-002 模型支持的输入最大 Token 数是8191，向量维度 1536
        # 在我们的 DEMO 中过滤 Token 超过 8000 的文本
        max_tokens = 8000
        """
        encoding = tiktoken.get_encoding(embedding_encoding)
        # 计算每条评论的token数量。我们通过使用encoding.encode方法获取每条评论的token数，然后把结果存储在新的'n_tokens'列中。
        df[self.DataFrameColumnNToken] = df[self.DataFrameColumnCombined].apply(lambda x: len(encoding.encode(x)))
        # 实际生成会耗时几分钟，逐行调用 OpenAI Embedding API
        df[self.DataFrameColumnEmbedding] = df[self.DataFrameColumnCombined].apply(embedding_function)
        return df


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Embedding Helper')
        self.parser.add_argument('--make_embedding_csv', type=str,
                                 help='需要进行csv格式化的数据路径.诸如origin_data/car_sales.csv')
        self.parser.add_argument('--test_load_faiss', action='store_true',
                                 help='将格式化的csv文件加载到faiss中, 并进行相似度搜索,测试faiss是否正常工作',
                                 default=False)

    def parse_arguments(self):
        args = self.parser.parse_args()
        return args


if __name__ == '__main__':
    def test_embedding_by_origin_file():
        """
        测试EmbeddingHelper.读取知识库, 并生成知识库的embedding
        :return:
        """
        # 1. 读取csv文件
        csv_file_path = "origin_data/car_sales.csv"
        # 2. 实例化OpenAIEmbedding
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        openai_embedding = OpenAIEmbedding(openai_api_key)
        # 3. 实例化EmbeddingHelper
        embedding_helper = EmbeddingHelper()
        # 4. 运行EmbeddingHelper
        print(f"Start embedding, input file: {csv_file_path}")
        embedding_helper.embedding_by_origin_file(csv_file_path, OpenAIEmbedding.TickTokenEncoding,
                                                  openai_embedding.embedding_text)
        # 5. 输出csv文件
        output_csv_path = "data/car_sales.csv"
        embedding_helper.output(output_csv_path)
        print(f"Embedding finished, output file: {output_csv_path}")
        print(embedding_helper.df.head(2))
        print(embedding_helper.df[EmbeddingHelper.DataFrameColumnNToken])


    def test_read_already_embedding_file():
        """
        测试EmbeddingHelper.读取已经embedding的csv文件
        :return:
        """
        # 1. 读取csv文件
        csv_file_path = "data/car_sales.csv"
        # 2. 实例化EmbeddingHelper
        embedding_helper = EmbeddingHelper()
        # 3. 运行EmbeddingHelper
        embedding_helper.read_already_embedding_file(csv_file_path)
        print(embedding_helper.df.head(2))


    # test_read_already_embedding_file()
    def load_faiss():
        """
        根据知识库加载faiss
        :return:
        """
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import OpenAIEmbeddings
        csv_file_path = "data/car_sales.csv"
        # 2. 实例化EmbeddingHelper
        embedding_helper = EmbeddingHelper()
        # 3. 运行EmbeddingHelper
        embedding_helper.read_already_embedding_file(csv_file_path)
        texts = embedding_helper.df[EmbeddingHelper.DataFrameColumnCombined].tolist()
        text_embeddings = embedding_helper.df[EmbeddingHelper.DataFrameColumnEmbedding].tolist()
        embeddings = OpenAIEmbeddings()
        text_embedding_pairs = zip(texts, text_embeddings)
        db = FAISS.from_embeddings(text_embedding_pairs, embeddings)
        # db.save_local("data/faiss")
        # query = "电池续航怎么样"
        # 在 Faiss 中进行相似度搜索，找出与 query 最相似结果
        # 实例化一个 similarity_score_threshold Retriever
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.8}
        )
        while True:
            query = input("客户问题: ")
            docs = retriever.get_relevant_documents(query)
            for doc in docs:
                print(doc.page_content + "\n")


    # 只需要运行一次. 测试EmbeddingHelper.读取知识库, 并生成知识库的embedding
    ##  test_embedding_by_origin_file()

    # 加载Faiss并进行相似度回答
    # load_faiss()
    arg_parser = ArgumentParser()
    args = arg_parser.parse_arguments()
    # print(args)
    if args.make_embedding_csv:
        test_embedding_by_origin_file()

    if args.test_load_faiss:
        load_faiss()
