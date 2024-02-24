import requests
import simplejson
import traceback
import time
import os
import openai

from model import Model
from utils import LOG
from openai import OpenAI


class OpenAIModel(Model):
    def __init__(self, model: str, api_key: str):
        self.model = model
        if os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.client = OpenAI(api_key=api_key)

    def make_request(self, prompt):
        attempts = 0
        while attempts < 3:
            try:
                if self.model == "gpt-3.5-turbo":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    translation = response.choices[0].message.content.strip()
                else:
                    response = self.client.completions.create(
                        model=self.model,
                        prompt=prompt,
                        max_tokens=150,
                        temperature=0
                    )
                    translation = response.choices[0].text.strip()

                return translation, True
            except openai.RateLimitError as e:
                attempts += 1
                if attempts < 3:
                    LOG.warning("Rate limit reached. Waiting for 60 seconds before retrying.")
                    time.sleep(60)
                else:
                    raise Exception("Rate limit reached. Maximum attempts exceeded.")
            except openai.APIConnectionError as e:
                LOG.error("The server could not be reached")
                LOG.error(
                    e.__cause__)  # an underlying Exception, likely raised within httpx.            except requests.exceptions.Timeout as e:
            except openai.APIStatusError as e:
                LOG.error("Another non-200-range status code was received")
                LOG.error(e.status_code)
                LOG.error(e.response)
            except Exception as e:
                LOG.error(traceback.format_exc())
                raise Exception(f"发生了未知错误：{e}")
        return "", False
