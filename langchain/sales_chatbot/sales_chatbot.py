import gradio as gr
import argparse
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from FaissHelper import FaissHelper


class AIAssistant(object):
    # SUPPORTED_ASSISTANT_TYPE_PROPERTY = "房产销售"
    SUPPORTED_ASSISTANT_TYPE_CAR = "汽车销售"
    SUPPORTED_ASSISTANT_TYPES = [SUPPORTED_ASSISTANT_TYPE_CAR]

    def __init__(self, vector_store_dir: str = "data/car/faiss", model_name: str = "gpt-3.5-turbo"):
        self.vector_store_dir = vector_store_dir
        self.sales_bot = self._initialize_sales_bot(vector_store_dir, model_name)

    @staticmethod
    def _initialize_sales_bot(vector_store_dir: str, model_name: str):
        embeddings = OpenAIEmbeddings()
        faiss = FaissHelper()
        faiss.load_local(vector_store_dir, embeddings, 0.8)

        llm = ChatOpenAI(model_name=model_name, temperature=0)
        bot = RetrievalQA.from_chain_type(llm, retriever=faiss.retriever)
        # 返回向量数据库的检索结果
        bot.return_source_documents = True

        return bot


# 定义全局小助理
assistant: AIAssistant


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True
    ans = assistant.sales_bot({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"


def init_sales_bot(faiss_dir: str):
    # 如果支持多种助理类型，可以在这里初始化
    global assistant
    assistant = AIAssistant(faiss_dir)


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='小助理')
        self.parser.add_argument('--faiss_dir', type=str, help='保存faiss文件的目录路径')

    def parse_arguments(self):
        args = self.parser.parse_args()
        if not args.faiss_dir:
            self.parser.error("--faiss_dir is required")
        return args


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="小助理",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )
    # 如果需要使用gradio的隧道打洞对外服务,开启share=True. 不对外展示关闭调试更佳. 对外打洞时server_name需要设置为"0.0.0.0"
    demo.launch(share=False, server_name="127.0.0.1")


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    args = arg_parser.parse_arguments()

    # 初始销售机器人
    init_sales_bot(args.faiss_dir)
    # 启动 Gradio 服务
    launch_gradio()
