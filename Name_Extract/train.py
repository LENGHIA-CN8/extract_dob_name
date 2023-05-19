import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset,DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from datasets import Dataset

from transformers import(
    AdamW,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

train_path = './UIT-ViNames-Dataset/UIT-ViNames/Train.csv'
val_path = './UIT-ViNames-Dataset/UIT-ViNames/Val.csv'
test_path = './UIT-ViNames-Dataset/UIT-ViNames/Test.csv'
null_path = './UIT-ViNames-Dataset/UIT-ViNames/Null.csv'
null_path_1 = './UIT-ViNames-Dataset/UIT-ViNames/Null_1.csv'


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["inputs"], max_length=256, truncation=True, padding=True
    )
    
    
    labels = tokenizer(
        examples["labels"], max_length=256, truncation=True, padding=True
    )
    model_inputs['labels'] = labels['input_ids']
    model_inputs['input_ids'] = model_inputs['input_ids']
    return model_inputs

def get_device_and_set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device



if __name__ == '__main__':
    SEED = 123
    device = get_device_and_set_seed(SEED)
    model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base") 
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    null_df = pd.read_csv(null_path)
    null_df_1 = pd.read_csv(null_path_1)
    train_df = pd.concat([train_df, null_df, null_df_1], ignore_index = False)
    
    dict_obj = {'inputs': train_df['sentence'], 'labels': train_df['Full_Names']}
    dataset = Dataset.from_dict(dict_obj)
    train_data = dataset.map(preprocess_function, batched=True, remove_columns=['inputs'], num_proc=8)
    
    dict_obj = {'inputs': test_df['sentence'], 'labels': test_df['Full_Names']}
    dataset = Dataset.from_dict(dict_obj)
    test_data = dataset.map(preprocess_function, batched=True, remove_columns=['inputs'], num_proc=8)

    dict_obj = {'inputs': val_df['sentence'], 'labels': val_df['Full_Names']}
    dataset = Dataset.from_dict(dict_obj)
    val_data = dataset.map(preprocess_function, batched=True, remove_columns=['inputs'], num_proc=8)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")


    training_args = Seq2SeqTrainingArguments("tmp/",
                                      do_train=True,
                                      do_eval=True,
                                      evaluation_strategy='steps',
                                      eval_steps=2475,
                                      num_train_epochs=15,
                                      learning_rate=1e-5,
                                      warmup_ratio=0.05,
                                      weight_decay=0.01,
                                      per_device_train_batch_size=4,
                                      per_device_eval_batch_size=4,
                                      logging_dir='./log',
                                      group_by_length=True,
                                      load_best_model_at_end=True,
                                      save_steps=2475,
                                      save_total_limit=1,
                                      #eval_steps=1,
                                      #evaluation_strategy="steps",
                                      # evaluation_strategy="no",
                                      fp16=True,
                                      )
    trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
    )

    trainer.train()