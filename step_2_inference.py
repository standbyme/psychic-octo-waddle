from transformers import pipeline


if __name__ == "__main__":
    model_id_list = ["meta-llama/Meta-Llama-3-8B", "microsoft/Phi-3-mini-4k-instruct"]
    prompt_list = [
        "How to make a cake",
        "How to make a pie",
        "How to eat a banana",
    ]
    result = []

    for model_id in model_id_list:
        pipe = pipeline("text-generation", model=model_id, device_map="auto")
        result_for_model = []
        for prompt in prompt_list:
            v = pipe(prompt)
            assert len(v) == 1
            result_for_model.append(v[0])
        result.append(result_for_model)
    
    print(result)