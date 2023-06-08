#!/usr/bin/env python
# coding: utf-8


import os
import json
import numpy as np
from tqdm import trange

from load_dataset import Text2SQLDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from tokenizers import AddedToken
from torch.utils.tensorboard import SummaryWriter
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# configuration = GPT2Config()
# configuration

# FOR PRINTING INTERMEDIATE TORCH SIZES
DEBUG_FLAG = False

# Define model
class EncoderDecoder(nn.Module):
    def __init__(self, gpt2_hidden_size, t5_hidden_size, max_input_length, 
                 max_output_length, gpt2_model, t5_model, batch_size, gpt2_tokenizer):
        super(EncoderDecoder, self).__init__()
        
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.t5_hidden_size = t5_hidden_size
        self.batch_size = batch_size
        
        self.gpt2 = GPT2Model.from_pretrained(gpt2_model)
        self.gpt2.resize_token_embeddings(len(gpt2_tokenizer))
        
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model)
        self.linear = nn.Linear(gpt2_hidden_size, t5_hidden_size)
    
        self.t5.config.is_encoder_decoder = False
    

    def forward(self, input_ids, input_mask,
                decoder_input_ids, decoder_attention_mask):
        
        # Encode input with GPT2
        gpt2_output = self.gpt2(input_ids=input_ids, attention_mask=input_mask)
        gpt2_output = gpt2_output.last_hidden_state
        
        if DEBUG_FLAG: print(f"gpt2_output - {gpt2_output.size()}")
        
        # Transform GPT2 output to T5 input shape
        t5_input = self.linear(gpt2_output)
        if DEBUG_FLAG: print(f"gpt2_output linear - {t5_input.size()}")
        
#         t5_input = t5_input.unsqueeze(1).repeat(1, self.max_output_length, 1)
#         if DEBUG_FLAG: print(f"gpt2_output linear unsqueeze - {t5_input.size()}")
        
#         t5_input = t5_input.view(self.batch_size, self.max_input_length, self.t5_hidden_size)
        t5_input = t5_input.unsqueeze(0)
        if DEBUG_FLAG: print(f"t5_input - {t5_input.size()}")
        
#         t5_outputs = self.t5(decoder_input_ids=decoder_input_ids,
#                              decoder_attention_mask=decoder_attention_mask,
#                              encoder_outputs=t5_input
#                             )
#         if DEBUG_FLAG: print(f"t5_input logits - {(t5_outputs.logits).size()}")
#         return t5_outputs.logits
    
        t5_outputs = self.t5(labels=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask,
                             encoder_outputs=t5_input,
                             return_dict = True
                            )
        
        if DEBUG_FLAG: print(f"t5_input - {type(t5_outputs)}")
        return t5_outputs

    def predict(self, input_ids, input_mask, batch_size, t5_tokenizer):
        
        self.gpt2.eval()
        # Encode input with GPT2
        gpt2_output = self.gpt2(input_ids=input_ids, attention_mask=input_mask)
        gpt2_output = gpt2_output.last_hidden_state
        if DEBUG_FLAG: print(f"gpt2_output - {gpt2_output.size()}")
        
        # Transform BERT output to T5 input shape
        t5_input = self.linear(gpt2_output)
        if DEBUG_FLAG: print(f"gpt2_output linear - {t5_input.size()}")
            
#         t5_input = t5_input.view(batch_size, self.max_input_length, self.t5_hidden_size)
        t5_input = t5_input.unsqueeze(0)
        if DEBUG_FLAG: print(f"t5_input - {t5_input.size()} - {t5_input}")
        
        # Generate initial input for T5 decoder
        start_token = t5_tokenizer.pad_token_id
        
#         decoder_input_ids = torch.tensor([start_token] * batch_size).unsqueeze(0)
#         decoder_attention_mask = torch.tensor([1] * batch_size).unsqueeze(0)
        
        decoder_input_ids = torch.tensor([start_token]*batch_size).unsqueeze(0)
        decoder_attention_mask = torch.tensor([1]*batch_size).unsqueeze(0)
    
#         decoder_input_ids = decoder_input_ids.view(decoder_input_ids.shape[1],
#                                                    decoder_input_ids.shape[0])
#         decoder_attention_mask = decoder_attention_mask.view(decoder_attention_mask.shape[1],
#                                                              decoder_attention_mask.shape[0])
        
        if DEBUG_FLAG: print(f"decoder_input_ids - {decoder_input_ids.size()}")
        if DEBUG_FLAG: print(f"decoder_attention_mask - {decoder_attention_mask.size()}")
        
        if DEBUG_FLAG: print(f"initial decoder_input_ids - {decoder_input_ids}")
        
        # Use the model to get output logits
        # Predict the output
        self.t5.eval()
        with torch.no_grad():
            for i in range(50):  # Maximum length of generated sequence
                t5_outputs = self.t5(decoder_input_ids=decoder_input_ids,
                                     decoder_attention_mask=decoder_attention_mask,
                                     encoder_outputs=t5_input)
#                 print(f"t5_outputs - {t5_outputs}")
                if DEBUG_FLAG: print(f"t5_outputs logits - {(t5_outputs.logits).size()}")
    
                next_token_logits = t5_outputs.logits[:, -1, :]
                if DEBUG_FLAG: print(f"next_token_logits - {next_token_logits.size()}")
            
#                 next_token_id = torch.argmax(next_token_logits, dim=-1)
                next_token_id = next_token_logits.argmax(1)
#                 print(f"next_token_id - {next_token_id.size()}")
#                 print(f"next_token_id.unsqueeze(-1) - {next_token_id.unsqueeze(-1).size()}")
                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.unsqueeze(-1)], dim=-1)
                decoder_attention_mask = torch.cat([decoder_attention_mask,
                                                    torch.ones_like(next_token_id.unsqueeze(-1))], dim=-1)

                if next_token_id == t5_tokenizer.eos_token_id:
                    break
                
                if DEBUG_FLAG: print(f"pred decoder_input_ids - {decoder_input_ids}")
                
#                 break
        
        # generated_text
#         t5_outputs = t5_tokenizer.decode(decoder_input_ids.squeeze(), skip_special_tokens=True)
        t5_outputs = decoder_input_ids #.squeeze()
        
        return t5_outputs


# Model class must be defined somewhere

model_folder = "models/GPT2_T5_lr0.0001_bs32_gpt2_t5-small/"
model_name = "model_60"

model_path = os.path.join(os.getcwd(), model_folder, model_name)
output = os.path.join(os.getcwd(), model_folder, f"{model_name}.txt")

model2 = torch.load(model_path, map_location=torch.device('cpu'))
model2.eval()


# In[39]:


import time
import tqdm
from text2sql_decoding_utils import decode_sqls
from spider_metric.evaluator import EvaluateTool

dev_filepath = "../data/resdsql_pre/preprocessed_dataset_test.json"
original_dev_filepath = "../data/split/spider_test.json"

batch_size = 1
max_input_length = 43
gpt2_model = 'gpt2'
t5_model = 't5-small'
db_path = "../spider_data/database"
mode = "eval"


# In[40]:


start_time = time.time()

# initialize tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model)

gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# if isinstance(tokenizer, T5TokenizerFast):
#     tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

dev_dataset = Text2SQLDataset(
            dir_ = dev_filepath,
            mode = mode)

dev_dataloder = DataLoader(
        dev_dataset, 
        batch_size = batch_size, 
        shuffle = False,
        collate_fn = lambda x: x,
        drop_last = False
    )

# initialize model

model2.eval()
predict_sqls = []
# for batch in tqdm(dev_dataloder):
for idx, batch in enumerate(dev_dataloder):
    batch_inputs = [data[0] for data in batch]
    batch_db_ids = [data[1] for data in batch]
    batch_tc_original = [data[2] for data in batch]

    tokenized_inputs = gpt2_tokenizer(batch_inputs,
                                      add_special_tokens=True,
                                      padding="max_length", #True,
                                      max_length=max_input_length,
                                      #pad_to_max_length=True,
                                      return_tensors='pt',
                                      truncation=True)

    encoder_input_ids = tokenized_inputs["input_ids"].to(device)
    encoder_input_attention_mask = tokenized_inputs["attention_mask"].to(device)
    
    # print(f"encoder_input_ids - {encoder_input_ids.size()}")
    # print(f"encoder_input_attention_mask - {encoder_input_attention_mask.size()}")

    with torch.no_grad():
        model_outputs = model2.predict(encoder_input_ids, encoder_input_attention_mask,
                                       batch_size, t5_tokenizer=t5_tokenizer)

        print(f"Actu - {batch_inputs[0]}")
        print(f"Pred - {t5_tokenizer.decode(model_outputs.squeeze(), skip_special_tokens=True)}")

        model_outputs = model_outputs.view(batch_size, 1, model_outputs.shape[1])
        
        predict_sqls += decode_sqls(
                                    db_path, 
                                    model_outputs, 
                                    batch_db_ids, 
                                    batch_inputs, 
                                    t5_tokenizer, 
                                    batch_tc_original
                                    )
    # break


new_dir = "/".join(output.split("/")[:-1]).strip()
if new_dir != "":
    os.makedirs(new_dir, exist_ok = True)

# save results
with open(output, "w", encoding = 'utf-8') as f:
    for pred in predict_sqls:
        f.write(pred + "\n")

end_time = time.time()
print("Text-to-SQL inference spends {}s.".format(end_time-start_time))

if mode == "eval":
    # initialize evaluator
    evaluator = EvaluateTool()
    evaluator.register_golds(original_dev_filepath, db_path)
    spider_metric_result = evaluator.evaluate(predict_sqls)

    em_res = spider_metric_result["exact_match"]
    ex_res = spider_metric_result["exec"]

    print('exact_match score: {}'.format(em_res))
    print('exec score: {}'.format(ex_res))
    
    
#     return spider_metric_result["exact_match"], spider_metric_result["exec"]
