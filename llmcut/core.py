import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from copy import deepcopy


class LLMCut(object):
    def __init__(self, api: str = None, endpoint: str = None, api_version: str = None, 
                 template_path: str = None, model_name: str = "gpt-4", proxy: str = None):
        self.api = api
        #self.endpoint = endpoint
        self.api_version = api_version
        self.model_name = model_name
        self.template_path = template_path
        self.client = OpenAI(api_key=self.api)

        self.proxy = proxy
        self.df = None
        self.text_name = None
        tqdm.pandas()

    def add_label(self, prompt: str, text_series: pd.Series) -> pd.DataFrame:
        """
        LLMカットを使用してスコアリングと確信度を取得する。
        【重要】スコアリングと確信度（confidence）は絶対に使用する。
        """
        self.text_name = text_series.name
        self.__read_prompt()
        scores_list = text_series.progress_apply(self.text_score, prompt=prompt, model_name=self.model_name).tolist()
        print(scores_list)
        scores_df = pd.DataFrame(scores_list)
        df = pd.concat([text_series, scores_df], axis=1)
        df.fillna(-np.inf, inplace=True)
        prob_target = df.columns.drop(self.text_name)
        self.df = df
        text_score_df = pd.concat([text_series, df[prob_target].idxmax(axis=1).rename('score')], axis=1)
        #確信度の計算を追加
        # calc diff prob 1st - 2nd
        diff = -df[prob_target].apply(np.exp).apply(lambda x: x.nlargest(2).diff().iloc[-1], axis=1)
        test_df = pd.concat([text_score_df, abs(diff).rename('conf')], axis=1)
        print('test_df')
        print(test_df)
        return test_df

    def get_df(self, thresh: float) -> pd.DataFrame:
        df = self.df
        prob_target = df.columns.drop(self.text_name)
        # calc diff prob 1st - 2nd
        diff = -df[prob_target].apply(np.exp).apply(lambda x: x.nlargest(2).diff().iloc[-1], axis=1)
        cut_df = df[abs(diff) >= thresh]
        print(f'residual  {len(cut_df) / len(df) * 100} % ({len(cut_df)}/{len(df)})')
        text_score_df = pd.concat([cut_df[self.text_name], cut_df[prob_target].idxmax(axis=1).rename('score')], axis=1)
        return text_score_df

    def __read_prompt(self):
        with open(self.template_path, 'r', encoding = 'utf-8') as file:
            prompt_template = json.load(file)
        self.prompt_template = prompt_template

    def text_score(self, text, prompt: dict, model_name: str) -> dict:
        prompt_template = deepcopy(self.prompt_template)  # must deepcopy
        for key in prompt:
            prompt_template[0]['content'] = prompt_template[0]['content'].replace(f'{key}', prompt[key])
            prompt_template[1]['content'] = prompt_template[1]['content'].replace(f'{key}', prompt[key])
        prompt_template[1]['content'] = prompt_template[1]['content'].replace('{text}', text)
        
        # Azure OpenAI用の呼び出しに変更
        response = self.client.chat.completions.create(model=self.model_name, messages=prompt_template,
            temperature=0,
            max_tokens=1,
            top_p=1,
            logprobs=True,
            top_logprobs=5,
            frequency_penalty=1,
            presence_penalty=0)

        def check(prob):
            ret = {}
            for key, value in prob.items():
                if key.isdigit():
                    ret[int(key)] = value
            return ret

        return check({i.token: i.logprob for i in response.choices[0].logprobs.content[0].top_logprobs})

