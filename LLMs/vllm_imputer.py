from vllm import LLM, SamplingParams
import gc
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
import torch
import time


class VLLMImputer():
    def __init__(self):
        # To use vllm for inference, you need to download the model files either using HuggingFace model hub or manually.
         # You should modify the path to the model according to your local environment.
        self.path_to_model = ("LLMs/downloaded_models/NECOUDBFM/Jellyfish")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LLM(model=self.path_to_model, trust_remote_code=True).to(self.device)
        # You can modify the sampling parameters according to your needs.
        # Caution: The stop parameter should not be changed.
        self.sampling_params = SamplingParams(
            temperature=0.35,
            top_p=0.9,
            max_tokens=1024,
            stop=["### Instruction:"],
        )
    
    def generate():
        print("Lets go")
        system_message = "You are an AI assistant that follows instruction extremely well. Help as much as you can."

        # You need to define the user_message variable based on the task and the data you want to test on.
        user_message = "You are presented with a mental health record that is missing a specific attribute: Suicide. Your task is to deduce or infer the value of Suicide using the available information in the record."

        prompt = f"{system_message}\n\n### Instruction:\n\n{user_message}\n\n### Response:\n\n"
        outputs = model.generate(prompt, sampling_params)
        response = outputs[0].outputs[0].text.strip()
        print(response)









