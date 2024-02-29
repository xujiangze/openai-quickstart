import sys
import os
import gradio as gr
import gradio.components as gr_comp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import ArgumentParser, ConfigLoader, LOG, check_argument
from model import GLMModel, OpenAIModel
from translator import PDFTranslator

GLOBAL_OPENAI_MODEL = None


def init_openai_model(model_name: str, api_key: str):
    global GLOBAL_OPENAI_MODEL
    GLOBAL_OPENAI_MODEL = OpenAIModel(model=model_name, api_key=api_key)


class WebServer(object):
    def __init__(self):
        pass

    @classmethod
    def process_pdfs(cls, pdf_files, trans_format, target_language, output_file_path, pages: int = 1):
        """
        批量翻译 PDF 文件
        :param pdf_files: 传入的pdf文件, 支持多个
        :param trans_format: 转化的格式
        :param target_language: 目标语言
        :param output_file_path: 输出路径
        :param pages: 翻译的页码
        :return:
        """
        results = []
        LOG.info(f"pdf_files={pdf_files}, trans_format={trans_format}, target_language={target_language}, "
                 f"output_file_path={output_file_path}, pages={pages}")
        for pdf_file in pdf_files:
            result = cls.translate_pdf(pdf_file.name, trans_format, target_language, output_file_path, pages)
            LOG.info(f"Result: {result}")
            results.append(result)
        return results

    @classmethod
    def translate_pdf(cls, pdf_file_path, trans_format, target_language, output_file_path, pages: int):
        translator = PDFTranslator(GLOBAL_OPENAI_MODEL)
        translator.translate_pdf(pdf_file_path, trans_format, target_language, output_file_path, pages)

        return "翻译完毕"

    def run(self):
        input_interface = [
            gr.Files(label="请添加多个需要翻译的pdf文件", file_count="multiple"),
            gr.Dropdown(label="生成格式", choices=["PDF", "markdown"], value="PDF"),
            gr.Textbox(label="翻译为 (e.g., 中文)", value="中文"),
            gr.Textbox(label="文档生成位置:eg: /tmp/translated.pdf", value="/tmp/translated.pdf"),
            gr.Number(label="翻译几页", value=1),
        ]

        output_interface = gr.Textbox(label="翻译结果")
        interface = gr.Interface(fn=self.process_pdfs, inputs=input_interface, outputs=output_interface)
        interface.launch()


if __name__ == "__main__":
    # 解析命令行参数
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()

    # 加载配置文件
    config_loader = ConfigLoader(args.config)
    config = config_loader.load_config()

    # 命令行参数优先级高于配置文件
    model_name = args.openai_model if args.openai_model else config[ArgumentParser.MODE_TYPE_OPENAI]['model']
    api_key = args.openai_api_key if args.openai_api_key else config[ArgumentParser.MODE_TYPE_OPENAI]['api_key']

    # 参数检查
    check_argument(args, model_name, api_key)

    # 初始化模型
    init_openai_model(model_name, api_key)

    # 启动web程序
    web_server = WebServer()
    web_server.run()
