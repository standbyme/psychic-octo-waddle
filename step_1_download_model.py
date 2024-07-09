from transformers import pipeline


# Download the model
def download_model(model_id):
    try:
        pipeline(
            "text-generation", model=model_id, device_map="auto", return_full_text=False
        )
    except Exception as e:
        pass


if __name__ == "__main__":
    model_id_list = ["meta-llama/Meta-Llama-3-8B", "microsoft/Phi-3-mini-4k-instruct"]
    for model_id in model_id_list:
        download_model(model_id)
