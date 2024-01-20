import pandas as pd
from transformers import AutoTokenizer, pipeline, AutoModelForTableQuestionAnswering, TapasForQuestionAnswering


class LLM:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
        self.model = AutoModelForTableQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

    def tokenize(self, table, queries):
        return self.tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")

    def answer(self, table, queries):
        inputs = self.tokenize(table, queries)
        outputs = self.model(**inputs)
        logits = outputs.logits.detach()
        # Convert logits to answer coordinates
        predicted_answer_coordinates, = self.tokenizer.convert_logits_to_predictions(inputs, logits)
        # Format the answer
        answers = []
        for coordinates in predicted_answer_coordinates:
            if len(coordinates) == 1:
                # Single cell answer
                answers.append(table.iat[coordinates[0]])
            else:
                # Multiple cell answer, join cells
                answer_cells = [table.iat[coord] for coord in coordinates]
                answers.append(", ".join(answer_cells))
        print(answers)
        return answers


batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]

# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
#
# model = AutoModelForCausalLM.from_pretrained(model_id)
#
# text = "Hello my name is"
# inputs = tokenizer(text, return_tensors="pt")
#
# outputs = model.generate(**inputs, max_new_tokens=20)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
#
#
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# torch.set_default_device("cuda")
#
# model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
#
# inputs = tokenizer('''def print_prime(n):
#    """
#    Print all primes between 1 and n
#    """''', return_tensors="pt", return_attention_mask=False)
#
# outputs = model.generate(**inputs, max_length=200)
# text = tokenizer.batch_decode(outputs)[0]
# print(text)
#
#
#
# from optimum.intel import OVModelForCausalLM
# from transformers import AutoTokenizer,
#
#   model_id = "helenai/gpt2-ov"
#  model = OVModelForCausalLM.from_pretrained(model_id)
#   tokenizer = AutoTokenizer.from_pretrained(model_id)
#   pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
#   results = pipe("He's a dreadful magician and")
