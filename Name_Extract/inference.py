import torch
from datasets import load_metric
from transformers import(
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
import argparse
import re
max_target_length = 256
stop_words = ['ừ','ơ','à','ờ','ạ']

def process_raw_text(text):
    pattern = r'\b(?:' + '|'.join(stop_words) + r')\b'
    return re.sub(pattern, '', text)

def get_prediction(text, model, tokenizer, device):
    text = process_raw_text(text)
    tokenized_text = tokenizer(text, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
          input_ids=tokenized_text['input_ids'].to(device),
          max_length=max_target_length,
          attention_mask=tokenized_text['attention_mask'].to(device),
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return result

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
    print(get_prediction(args.text, model, tokenizer, args.device))
    
if __name__ == '__main__':
    main()

    
    
    