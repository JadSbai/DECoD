from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd


class MentalHealthImputer:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    def impute_data(self, table, clinician_prior):
        # Convert the table to a text representation
        table_text = self.table_to_text(table)
        max_new_tokens = 100
        prompt = (
            f"Task: Impute the missing values in the mental health dataset below. "
            f"Use the provided clinician's insights to guide your imputations and "
            f"explain your reasoning for each imputation.\n\n"
            f"Clinician's Prior: {clinician_prior}\n\n"
            f"Dataset:\n{table_text}\n\n"
            "Impute the missing values and explain the reasoning:"
        )

        # Tokenize and generate output
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        print('tokenized')
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        print('generated')
        imputed_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(imputed_text)