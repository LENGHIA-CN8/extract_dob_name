# import sys
# sys.path.append('./Vietnamese_Inverse_Text_Normalization')
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from Vietnamese_Inverse_Text_Normalization.DOB import extract_dob
import yaml
from Name_Extract.inference import get_prediction
import logging
from transformers import(
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

logger = logging.getLogger(__name__)

with open("config.yaml") as f:
    config = yaml.safe_load(f)

logger.info('Loading model ...')
model = AutoModelForSeq2SeqLM.from_pretrained(config['model']).to(config['device'])
tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])

app = FastAPI()

class Textmessage(BaseModel):
    message: str

@app.get("/")
def main():
    return {"message": "Welcome!"}

@app.post("/extract_dob")
async def predict_dob(text_message: Textmessage):
    logger.info(f'input message: {text_message.message}')
    message = extract_dob(text_message.message)
    logger.info(f'output message: {message}')
    return {"DOB": message}

@app.post("/extract_name")
async def predict_name(text_message: Textmessage):
    logger.info(f'input message: {text_message.message}')
    message = get_prediction(text_message.message, model, tokenizer, config['device'])
    message = message.strip().title()
    if message == 'None':
        return {"Name": None}
    logging.info(f'output message: {message}')
    return {"Name": message}

if __name__ == "__main__":
    uvicorn.run(app, host="10.5.1.230", port=8002)