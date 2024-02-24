import argparse


class ArgumentParser:
    MODE_TYPE_GLM = 'GLMModel'
    MODE_TYPE_OPENAI = 'OpenAIModel'
    MODEL_TYPE_CHOICES = [MODE_TYPE_GLM, MODE_TYPE_OPENAI]

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Translate English PDF book to Chinese.')
        self.parser.add_argument('--config', type=str, default='config.yaml',
                                 help='Configuration file with model and API settings.')
        self.parser.add_argument('--model_type', type=str, required=True, choices=self.MODEL_TYPE_CHOICES,
                                 help='The type of translation model to use. '
                                      'Choose between "GLMModel" and "OpenAIModel".')
        self.parser.add_argument('--glm_model_url', type=str, help='The URL of the ChatGLM model URL.')
        self.parser.add_argument('--timeout', type=int, help='Timeout for the API request in seconds.')
        self.parser.add_argument('--openai_model', type=str,
                                 help='The model name of OpenAI Model. Required if model_type is "OpenAIModel".')
        self.parser.add_argument('--openai_api_key', type=str,
                                 help='The API key for OpenAIModel. Required if model_type is "OpenAIModel".')
        self.parser.add_argument('--book', type=str, help='PDF file to translate.')
        self.parser.add_argument('--file_format', type=str,
                                 help='The file format of translated book. Now supporting PDF and Markdown')

    def parse_arguments(self):
        args = self.parser.parse_args()
        return args


def check_argument(args, model_name: str, api_key: str):
    """
    检查参数是否正确
    :param args:
    :param model_name:
    :param api_key:
    :return:
    """
    if args.model_type == ArgumentParser.MODE_TYPE_OPENAI and (not model_name or not api_key):
        raise ValueError("--openai_model and --openai_api_key is required when using OpenAIModel")
