from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR", "CC_data")
ds = load_dataset("MMDocIR/MMDocIR-Challenge")
ds.save_to_disk("./datasets")
