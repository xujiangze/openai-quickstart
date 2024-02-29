import argparse
from typing import Tuple, List, Iterable
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document


class FaissHelper(object):
    """
    协助初始化Faiss
    """

    def __init__(self):
        self.db: FAISS = None
        self.retriever = None
        self.score_threshold: float = 0.8

    def new_by_text(
            self,
            text_embedding_pairs: Iterable[Tuple[str, List[float]]],
            embeddings: OpenAIEmbeddings,
            score_threshold: float = 0.8
    ):
        self.db: FAISS = FAISS.from_embeddings(text_embedding_pairs, embeddings)
        self.retriever = self.db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.8}
        )
        self.score_threshold: float = score_threshold

    def set_score_threshold(self, score_threshold: float):
        self.score_threshold = score_threshold
        self.retriever = self.db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": self.score_threshold}
        )

    def query(self, query: str) -> List[Document]:
        return self.retriever.get_relevant_documents(query)

    def save_to_local(self, dir_path: str):
        self.db.save_local(dir_path)

    def load_local(self, dir_path: str, embeddings: OpenAIEmbeddings, score_threshold: float):
        self.db = FAISS.load_local(dir_path, embeddings)
        self.score_threshold = score_threshold
        self.retriever = self.db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score_threshold}
        )


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='FaissHelper')
        self.parser.add_argument('--test_query', action='store_true',
                                 help='测试加载本地的向量文件,并初始化faiss进行query测试',
                                 default=False)
        self.parser.add_argument('--test_save', action='store_true',
                                 help='测试将已经保存的向量文件,转化为faiss保存到本地',
                                 default=False)
        self.parser.add_argument('--save_dir', type=str, help='保存faiss文件的路径')
        self.parser.add_argument('--test_load', action='store_true',
                                 help='测试将本地的faiss文件加载到内存,并进行query测试',
                                 default=False)

    def parse_arguments(self):
        args = self.parser.parse_args()
        return args


if __name__ == '__main__':
    def test_query():
        """
        测试FaissHelper.query
        :return:
        """
        from EmbeddingHelper import EmbeddingHelper
        csv_file_path = "data/car_sales.csv"
        # 2. 实例化EmbeddingHelper
        embedding_helper = EmbeddingHelper()
        embedding_helper.read_already_embedding_file(csv_file_path)
        text_embedding_pairs, embeddings = embedding_helper.get_text_embedding_pairs()
        faiss = FaissHelper()
        faiss.new_by_text(text_embedding_pairs, embeddings)
        while True:
            query = input("客户问题: ")
            docs = faiss.query(query)
            for doc in docs:
                print(doc.page_content + "\n")


    def test_save_local(save_dir):
        """
        测试读取源文件之后, 保存到本地的Faiss文件
        :return:
        """
        from EmbeddingHelper import EmbeddingHelper
        csv_file_path = "data/car_sales.csv"
        # 2. 实例化EmbeddingHelper
        embedding_helper = EmbeddingHelper()
        embedding_helper.read_already_embedding_file(csv_file_path)
        text_embedding_pairs, embeddings = embedding_helper.get_text_embedding_pairs()
        faiss = FaissHelper()
        faiss.new_by_text(text_embedding_pairs, embeddings)
        faiss.save_to_local(f"{save_dir}/faiss")


    def test_load_local():
        """
        测试加载本地Faiss文件,并进行测试
        :return:
        """
        from EmbeddingHelper import EmbeddingHelper
        local_dir = "data/faiss"
        embeddings = OpenAIEmbeddings()
        faiss = FaissHelper()
        faiss.load_local(local_dir, embeddings, 0.8)
        while True:
            query = input("客户问题: ")
            docs = faiss.query(query)
            for doc in docs:
                print(doc.page_content + "\n")


    arg_parser = ArgumentParser()
    args = arg_parser.parse_arguments()
    if args.test_query:
        test_query()
    if args.test_save:
        test_save_local(args.save_dir)
    if args.test_load:
        test_load_local()

    # 测试问题: 电池续航能力如何
