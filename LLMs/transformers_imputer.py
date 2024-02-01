# import comet_llm
# import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
import torch
import time
import model_ids


class TransformersImputer:
    def __init__(self, model_id, manager):
        torch.set_default_device("cuda")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, torch_dtype="auto", trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # The optimization tricks are only supported on GPU
        self.config = BitsAndBytesConfig(load_in_8bit = True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
        self.manager = manager


    def impute_data(self):
        prompt = self.manager.get_prompt(self.model_id)
        start_time = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        try:
            if self.model_id == model_ids.jellyfish:
                system_message = "You are an AI assistant that follows instruction extremely well. Help as much as you can."
                final_prompt = f"{system_message}\n\n### Instruction:\n\n{prompt}\n\n### Response:\n\n"
                input_ids = inputs["input_ids"].to(self.device)
                # You can modify the sampling parameters according to your needs.
                generation_config = GenerationConfig(
                    do_sample=True,
                    temperature=0.35,
                    top_p=0.9,
                )
                with torch.no_grad():
                    generation_output = self.model.generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=1024,
                        pad_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.15,
                    )
                    output = generation_output[0]
                    response = self.tokenizer.decode(
                        output[:, input_ids.shape[-1] :][0], skip_special_tokens=True
                    ).strip()
            else:
                inputs.to(self.device)
                raw_outputs = self.model.generate(**inputs, max_new_tokens=1024, do_sample=True)
                response = self.tokenizer.batch_decode(raw_outputs)[0]


            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")
            # comet_llm.log_prompt(
            #     prompt=prompt,
            #     output=output,
            #     metadata={"model": self.model_id},
            #     duration=execution_time,
            #     api_key="aoTlL1hYOBHc0tfqLPaEu9Z2n",
            # )
            print(response)
            return response
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        finally:
            # print(f"Memory usage: {psutil.virtual_memory().percent}%")
            print('lol')



