import base64
import pandas as pd
from urllib.request import urlopen
from transformers import pipeline
import pickle

file_dict = {
"bert.pkl": ["https://1drv.ms/u/s!AvGV6EHyFTNkg-tpkys2nYpcS9oGYw?e=oM9sbj", "https://1drv.ms/u/s!AvGV6EHyFTNkg-to8Tx5AFxEP-talw?e=gE5Vr9"]
}

def create_onedrive_directdownload (onedrive_link):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return resultUrl

model_url = create_onedrive_directdownload(file_dict["bert.pkl"][0])
tokenizer_url = create_onedrive_directdownload(file_dict["bert.pkl"][1])
print("bert file is at", model_url)
model_raw = urlopen(model_url)
tokenizer_raw = urlopen(tokenizer_url)

model = pickle.load(model_raw)
tokenizer = pickle.load(tokenizer_raw)

def run_bert(query):
    polarity_task = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
    query = list(pd.Series([query]))
    result = polarity_task(query)

    return result[0]['score'] # can return result[0]['LABEL'] for the label value too

print(run_bert("I love tsla so much!!"))