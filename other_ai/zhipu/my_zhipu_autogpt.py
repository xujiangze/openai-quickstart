import os
import time
import jwt
from zhipuai import ZhipuAI
from langchain.agents import tool
from typing import Callable, Any
from langchain_openai import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain_experimental.autonomous_agents import AutoGPT

import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore


zhipuai_client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))


class ZhipuAI(object):
    ZhipuAIEmbeddingModel = "embedding-2"
    ZhipuAIEmbeddingSize = 1024
    ZhipuAIModeName = "glm-4"
    ZhipuAIAPIBase = "https://open.bigmodel.cn/api/paas/v4"

    @classmethod
    def get_embedding_query(cls) -> Callable:
        def embedding_query(content):
            """
            获取文本的embedding
            :param content:
            :return:
            """
            response = zhipuai_client.embeddings.create(
                model=cls.ZhipuAIEmbeddingModel,  # 填写需要调用的模型名称
                input=content,
            )
            return response.data[0].embedding

        return embedding_query

    @staticmethod
    def generate_token(apikey: str, exp_seconds: int):
        try:
            id, secret = apikey.split(".")
        except Exception as e:
            raise Exception("invalid apikey", e)

        payload = {
            "api_key": id,
            "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
            "timestamp": int(round(time.time() * 1000)),
        }

        return jwt.encode(
            payload,
            secret,
            algorithm="HS256",
            headers={"alg": "HS256", "sign_type": "SIGN"},
        )

    @classmethod
    def get_glm(cls, temperature: float, api_key: str, expire_time: int = 3600):
        """
        获取glm模型
        :param temperature: 随机度(靠进0越准确)
        :param expire_time: token过期时间. 智谱AI的是用token进行交互.并非API Key.
        :param api_key: 质谱AI的apikey.申请地址https://open.bigmodel.cn/overview
        :return:
        """
        llm = ChatOpenAI(
            model_name=cls.ZhipuAIModeName,
            openai_api_base=cls.ZhipuAIAPIBase,
            openai_api_key=cls.generate_token(api_key, expire_time),
            # 如果是自动问答机器人.可能要试一下streaming=True
            streaming=False,
            temperature=temperature,
            verbose=True,
        )
        return llm

class MyTools(object):
    @staticmethod
    def get_search_tool() -> Tool:
        """
        外部Google搜索工具
        :return:
        """
        search = SerpAPIWrapper()
        return Tool(
            name="search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        )

    @staticmethod
    def get_write_file_tool() -> WriteFileTool:
        """
        写文件工具
        :return:
        """
        return WriteFileTool()

    @staticmethod
    def get_read_file_tool() -> ReadFileTool:
        """
        读文件工具
        :return:
        """
        return ReadFileTool()

    @staticmethod
    def get_word_length_tool() -> Callable[..., Any]:
        @tool
        def get_word_length(word: str) -> int:
            """自定义工具，获取单词长度"""
            return len(word)

        return get_word_length

    @classmethod
    def get_tools(cls):
        return [
            cls.get_search_tool(),
            cls.get_write_file_tool(),
            cls.get_read_file_tool(),
            cls.get_word_length_tool()
        ]


def get_vectorstore():
    """
    获取向量数据库
    :return:
    """
    embedding_size = ZhipuAI.ZhipuAIEmbeddingSize
    # 使用 Faiss 的 IndexFlatL2 索引
    index = faiss.IndexFlatL2(embedding_size)
    # 实例化 Faiss 向量数据库
    #  InMemoryDocstore({}) 用内存去存储
    # 最后一个参数是一个空字典，用于初始化向量数据库为空
    vectorstore = FAISS(ZhipuAI.get_embedding_query(), index, InMemoryDocstore({}), {})
    return vectorstore


def get_zhipu_auto_gpt_agent():
    tools = MyTools.get_tools()
    vectorstore = get_vectorstore()

    agent = AutoGPT.from_llm_and_tools(
        ai_name="Jarvis",
        ai_role="Assistant",
        tools=tools,
        llm=ZhipuAI.get_glm(0.1, api_key=os.getenv("ZHIPU_API_KEY")),
        memory=vectorstore.as_retriever(
            search_type="similarity_score_threshold",  # 携带相似度搜索的阈值
            search_kwargs={"score_threshold": 0.8}),  # 实例化 Faiss 的 VectorStoreRetriever
    )
    return agent


if __name__ == '__main__':
    agent = get_zhipu_auto_gpt_agent()
    # ret = agent.run(["2023年成都大运会，中国金牌数是多少"])
    # print(ret)
    ret = agent.run(["'xxxcode' 单词的长度是多少, 并且告诉我2023年成都大运会，中国金牌数是多少"])
    print(ret)
    print(ret)

