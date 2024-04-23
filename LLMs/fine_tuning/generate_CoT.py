import json

import comet_llm
import openai
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

from sklearn.model_selection import train_test_split

from insights import BirthInsights

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
project_id = os.getenv('OPENAI_PROJECT_ID')
org_id = os.getenv('OPENAI_ORG_ID')

openai.api_key = api_key


def calculate_api_call_cost(num_input_tokens, num_generated_tokens, price_per_1000_input_tokens=0.01,
                            price_per_1000_generated_tokens=0.03):
    input_cost = (num_input_tokens / 1000) * price_per_1000_input_tokens
    generated_cost = (num_generated_tokens / 1000) * price_per_1000_generated_tokens
    total_cost = input_cost + generated_cost
    print(f"Total cost: ${total_cost:.2f}")


class GPT4Generator:
    def __init__(self, data, targets):
        self.client = OpenAI(
            organization=org_id,
            project=project_id,
        )
        self.insights = BirthInsights(data)
        self.data = data
        self.targets = targets

        comet_llm.init(project="GPT4-Imputation-Reasoning", api_key="aoTlL1hYOBHc0tfqLPaEu9Z2n")

    def craft_prompt(self, row, target):
        target_insights, sub_insights, known_correlations = self.insights.get_full_insights(row, target)
        formatted_target_insights = "\n".join(f"- {insight} \n" for insight in target_insights)
        formatted_sub_insights = "\n".join(f"- {insight} \n" for insight in sub_insights)
        formatted_known_correlations = "\n".join(f"- {correlation}" for correlation in known_correlations)

        prompt_template = f"""
        You are tasked with analyzing clinical data to deduce/explain the value of the target variable {target} based on provided statistical insights and known scientific correlations. The objective is to generate a detailed, logical reasoning that explains the true value of the target variable. 
        
        ### Provided Data:
        - Patient Record: {row}
        - The true value of the target variable {target} is: {row[target]}
        
        ### Description of the Data:
        Use the following descriptions to make sense of the values in the patient record and the insights provided.
        {self.insights.description}
        

        ### Statistical Insights for the Target Variable:
        {formatted_target_insights}\n
        - Note: When analyzing these insights, consider the proportion with respect to the total as an important factor before drawing conclusions.
        
        ### Statistical Insights for the Subpopulations this patient belongs to:
        {formatted_sub_insights if formatted_sub_insights != "" else "None"} \n
        {"- Note: Correlations are not absolute, and multiple variables might have similar effects on the target variable. Check all correlations." if formatted_sub_insights != "" else ""}


        ### Known Correlations:
        {formatted_known_correlations if formatted_known_correlations != "" else "None"}

        ### Task:
        1. Review and analyze the provided insights. Reflect on how each insight might influence {target}, taking into account the proportion of each category.
        2. Identify significant correlations from the insights and the known correlations. Assess whether these correlations apply to the specific case given the entire context of the patient's data.
        3. Construct a concise and succinct logical reasoning for the given value of {target}, utilizing the specific values from the patient record, the deduced correlations, and the known correlations.
        4. Do not openly assume in your reasoning any knowledge of the true value of the target variable.
        5. Report concisely your output inside a JSON object with the following format:
        
        ### Expected Output Format:
        {{
          "value": {row[target]},
          "reasoning": "Based on [detailed logical steps that lead to the conclusion, using the patient data and insights].",
        }}
        """
        return prompt_template

    def upload_file(self, file_path):
        response = self.client.files.create(
            file=open(file_path, "rb"),
            purpose='batch'
        )
        print(response.id)
        return response.id

    def generate_fine_tuning_data(self):
        datasets = {}
        for target in self.targets:
            samples = self.diverse_sampling(target, 5)
            datasets[target] = samples

        print(datasets)

        for target, data in datasets.items():
            with open(f'{target}_requests.jsonl', 'w') as f:
                for index, row in data.iterrows():
                    message = self.craft_prompt(row.to_dict(), target)
                    messages = [
                        {"role": "system",
                         "content": "You are a helpful clinical assistant and expert data analyst designed to output JSON."},
                        {"role": "user", "content": message},
                    ]

                    request = {"custom_id": f"{target}_request_{index}", "method": "POST",
                               "url": "/v1/chat/completions",
                               "body": {"model": "gpt-4-turbo",
                                        "messages": messages,
                                        "response_format": {"type": "json_object"},
                                        "max_tokens": 300,
                                        "temperature": 0.2,
                                        "n": 1,
                                        "seed": 0
                                        }
                               }
                    # Write the JSON object to the file
                    f.write(json.dumps(request) + '\n')

    def create_batch(self, file_id):
        response = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print(response.id)

    def list_files(self):
        files = self.client.files.list()
        print(files.data)
        return files.data

    def get_batch_status(self, batch_id):
        return self.client.batches.retrieve(batch_id)

    def cancel_batch(self, batch_id):
        self.client.batches.cancel(batch_id)

    def list_batches(self):
        return self.client.batches.list()

    def retrieve_batch_results(self, batch_id):
        batches = self.client.batches.list().data
        for batch in batches:
            if batch.id == batch_id:
                content = self.client.files.content(batch.output_file_id)
                return content

    def diverse_sampling(self, target_variable, n_samples=50):
        # Bin the target variable into quantiles
        self.data['quantile'] = pd.qcut(self.data[target_variable], q=5, duplicates='drop', labels=False)

        # Sample uniformly across these quantiles
        sampled_data, _ = train_test_split(self.data, stratify=self.data['quantile'], train_size=n_samples)

        # Drop the quantile column after sampling
        sampled_data = sampled_data.drop(columns=['quantile'])
        return sampled_data

    def generate(self, message):
        messages = [
            {"role": "system",
             "content": "You are a helpful clinical assistant and expert data analyst designed to output JSON."},
            {"role": "user", "content": message},
        ]
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            response_format={"type": "json_object"},
            messages=messages,
            max_tokens=300,
            temperature=0.2,
            n=1,
            seed=0,
        )
        calculate_api_call_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
        output = json.loads(response.choices[0].message.content)

        assert 'value' in output, "JSON object does not contain 'value' field"
        assert 'reasoning' in output, "JSON object does not contain 'reasoning' field"

        return output
