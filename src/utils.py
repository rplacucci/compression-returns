def tokenize_fn(example, task_name, tokenizer):
    if task_name == "ag_news":
        return tokenizer(example["content"], truncation=True)
    elif task_name == "dbpedia_14":
        return tokenizer(example["text"], truncation=True)
    else:
        raise NotImplementedError(f"Task '{task_name}' is not supported")

def postprocess_fn(dataset, task_name):
    columns_to_remove = []
    if task_name == "ag_news":
        columns_to_remove.extend(["title", "content"])
    elif task_name == "dbpedia_14":
        columns_to_remove.extend(["text"])
    else:
        raise NotImplementedError(f"Task '{task_name}' is not supported")
    
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.remove_columns(columns_to_remove)
    dataset.set_format("torch")
    return dataset