from transformers import AutoModel, AutoTokenizer

def download_model(model_id):
    model = AutoModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.save_pretrained(f'LLMs/downloaded_models/{model_id}')  # saves the model
    tokenizer.save_pretrained(f'LLMs/downloaded_models/{model_id}')  # saves the tokenizer

