from huggingface_hub import InferenceClient
import comet_llm


class HostedMentalHealthImputer:
    def __init__(self, model_id, manager):
        self.client = InferenceClient(token='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')
        self.manager = manager
        self.model_id = model_id

    def impute_data(self):
        prompt = self.manager.get_prompt()
        print('ready!!')
        output = self.client.fill_mask(prompt, model=self.model_id)
        comet_llm.log_prompt(
            prompt=prompt,
            output=output,
            api_key="aoTlL1hYOBHc0tfqLPaEu9Z2n",
        )
        print(output)
        return output[0]['generated_text']


