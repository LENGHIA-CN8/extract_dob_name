# import sys
# sys.path.append('./Vietnamese_Inverse_Text_Normalization')
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks
from Vietnamese_Inverse_Text_Normalization.DOB import extract_dob
import yaml
from Name_Extract.inference import get_prediction
import logging
from transformers import(
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
import telegram

logger = logging.getLogger(__name__)

with open("config.yaml") as f:
    config = yaml.safe_load(f)

logger.info('Loading model ...')
model = AutoModelForSeq2SeqLM.from_pretrained(config['model']).to(config['device'])
tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])

bot = telegram.Bot(token=config['BOT_TOKEN'])


async def send_message_to_telegram(input_message: str, output_message: str):
    full_message = f"Input: {input_message}\nOutput: {output_message}"
    await bot.send_message(chat_id=config['CHAT_ID'], text=full_message)

app = FastAPI()

class Textmessage(BaseModel):
    message: str

@app.get("/")
def main():
    return {"message": "Welcome!"}

@app.post("/extract_dob")
async def predict_dob(background_tasks: BackgroundTasks, text_message: Textmessage):
    input_message = text_message.message
    logger.info(f'input message: {input_message}')
    message = extract_dob(input_message)
    logger.info(f'output message: {message}')
    background_tasks.add_task(send_message_to_telegram, input_message, message)
    return {"DOB": message}

@app.post("/extract_name")
async def predict_name(background_tasks: BackgroundTasks, text_message: Textmessage):
    input_message = text_message.message
    logger.info(f'input message: {input_message}')
    message = get_prediction(input_message, model, tokenizer, config['device'])
    message = message.strip().title()
    logging.info(f'output message: {message}')
    background_tasks.add_task(send_message_to_telegram, input_message, message)
    if message == 'None':
        return {"Name": None}
    return {"Name": message}

if __name__ == "__main__":
    uvicorn.run(app, host="10.5.1.230", port=8002)