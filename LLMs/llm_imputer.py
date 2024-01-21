import comet_llm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


class MentalHealthImputer:
    def __init__(self, model_id, manager):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # The optimization tricks are only supported on GPU
        self.config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True,
                                                          attn_implementation="flash_attention_2", device_map="auto",
                                                          quantization_config=self.config)
        # The optimization tricks are only supported on GPU
        self.model.to(self.device)
        self.manager = manager

    def impute_data(self):
        prompt = self.manager.get_prompt()
        print('ready!!')
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=1000)
        output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        comet_llm.log_prompt(
            prompt=prompt,
            output=output,
            api_key="aoTlL1hYOBHc0tfqLPaEu9Z2n",
        )
        print(output)
        return output
