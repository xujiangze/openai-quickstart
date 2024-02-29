import sys
import os
import gradio as gr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import ArgumentParser, LOG
from translator import PDFTranslator, TranslationConfig


def translation(input_file, source_language, target_language, target_style):
    LOG.debug(f"[翻译任务]\n"
              f"源文件: {input_file.name}\n"
              f"源语言: {source_language}\n"
              f"目标语言: {target_language}\n"
              f"目标风格: {target_style}")

    output_file_path = Translator.translate_pdf(
        input_file.name,
        source_language=source_language,
        target_language=target_language,
        target_style=target_style
    )
    LOG.debug(f"[翻译结果] 文件位置:{output_file_path}\n")
    return output_file_path


def launch_gradio():
    iface = gr.Interface(
        fn=translation,
        title="OpenAI-Translator v2.0(PDF 电子书翻译工具)",
        inputs=[
            gr.File(label="上传PDF文件"),
            gr.Textbox(label="源语言（默认：英文）", placeholder="English", value="English"),
            gr.Textbox(label="目标语言（默认：中文）", placeholder="Chinese", value="Chinese"),
            gr.Textbox(label="翻译风格 (默认: 童话)", placeholder="fairy tale", lvalue="fairy tale"),
        ],
        outputs=[
            gr.File(label="下载翻译文件")
        ],
        allow_flagging="never"
    )

    # 如果需要使用gradio的隧道打洞对外服务,开启share=True. 不对外展示关闭调试更佳. 对外打洞时server_name需要设置为"0.0.0.0"
    iface.launch(share=False, server_name="127.0.0.1")


def initialize_translator():
    # 解析命令行
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()

    # 初始化配置单例
    config = TranslationConfig()
    config.initialize(args)    
    # 实例化 PDFTranslator 类，并调用 translate_pdf() 方法
    global Translator
    Translator = PDFTranslator(config.model_name)


if __name__ == "__main__":
    # 初始化 translator
    initialize_translator()
    # 启动 Gradio 服务
    launch_gradio()
