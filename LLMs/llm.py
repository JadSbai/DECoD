import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class LLM:
    def __init__(self, model_name='phi3', seed=0):
        torch.random.manual_seed(seed)
        self.model = AutoModelForCausalLM.from_pretrained(
            f"./llm_models/{model_name}",
            local_files_only=True,
        ).to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(f"./llm_tokenizers/{model_name}")

    def zero_shot(self, prompt):
        messages = [{'role': 'user', 'content': prompt}]
        return self.impute(messages)

    def impute(self, messages):
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('cuda')
        beams = 2
        generation_args = {
            "max_new_tokens": 800,
            "temperature": 0.2,
            "do_sample": True,
            "num_beams": beams,
            "num_beam_groups": 2,
            "diversity_penalty": 1.5,
            "renormalize_logits": True,
            "repetition_penalty": 1.15,
            "length_penalty": 1.0,
        }
        start = time.time()
        print('Started!')
        outputs = self.model.generate(inputs, **generation_args)
        text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        total_input_length = sum(len(message) for message in messages[:-1])
        output = text[total_input_length:].strip()
        print('Total time taken in seconds:', time.time() - start)
        return output
