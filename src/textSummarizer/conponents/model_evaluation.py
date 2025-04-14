from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import evaluate
import torch
import pandas as pd
from tqdm import tqdm
from textSummarizer.entity import ModelEvaluationConfig



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """Split the dataset into smaller batches that we can process simultaneously."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer, 
                                   batch_size=2, device="cpu",  # Forced to CPU
                                   column_text="dialogue", 
                                   column_summary="summary"):
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):
            inputs = tokenizer(article_batch, max_length=512, truncation=True, 
                             padding="max_length", return_tensors="pt")
            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8, num_beams=4, max_length=128
            )
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                               clean_up_tokenization_spaces=True) 
                               for s in summaries]
            decoded_summaries = [d.strip() for d in decoded_summaries]  # Clean up whitespace
            metric.add_batch(predictions=decoded_summaries, references=target_batch)
        
        score = metric.compute()
        return score

    def evaluate(self):
        device = "cpu"  # Forced to CPU
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
            print(f"Model and tokenizer loaded successfully on {device}")
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            raise

        # Loading data with fallback to CSV if dataset not found
        try:
            dataset_samsum_pt = load_from_disk(self.config.data_path)
            print("Dataset loaded successfully from disk")
        except FileNotFoundError:
            print("Processed dataset not found. Loading from CSV as fallback.")
            try:
                df_train = pd.read_csv("artifacts/data_ingestion/samsum_dataset/samsum-train.csv")
                df_validation = pd.read_csv("artifacts/data_ingestion/samsum_dataset/samsum-validation.csv")
                df_test = pd.read_csv("artifacts/data_ingestion/samsum_dataset/samsum-test.csv")
                required_columns = ["dialogue", "summary"]
                for df in [df_train, df_validation, df_test]:
                    if not all(col in df.columns for col in required_columns):
                        raise ValueError(f"CSV missing required columns: {required_columns}")
                from datasets import Dataset
                dataset = Dataset.from_pandas(pd.concat([df_train, df_validation, df_test])).train_test_split(test_size=0.2)
                dataset_samsum_pt = dataset
                print("Dataset created from CSV files")
            except Exception as e:
                print(f"Error loading CSV files: {e}")
                raise

        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_metric = evaluate.load("rouge")

        score = self.calculate_metric_on_test_ds(
            #dataset_samsum_pt['test'][0:10], rouge_metric, model_pegasus, tokenizer,
            dataset_samsum_pt['test'], rouge_metric, model_pegasus, tokenizer,
            batch_size=2, column_text="dialogue", column_summary="summary"
        )

        # Updated to use direct float values instead of .mid.fmeasure
        rouge_dict = dict((rn, score[rn]) for rn in rouge_names)
        df = pd.DataFrame(rouge_dict, index=["pegasus"])
        df.to_csv(self.config.metric_file_name, index=False)
        print(f"Evaluation results saved to {self.config.metric_file_name}")