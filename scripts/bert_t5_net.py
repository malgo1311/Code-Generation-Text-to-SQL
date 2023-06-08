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
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import BertModel, T5ForConditionalGeneration, T5Tokenizer, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device detected as - {device}")

# FOR PRINTING INTERMEDIATE TORCH SIZES
DEBUG_FLAG = False

# Define model
class EncoderDecoder(nn.Module):
    def __init__(self, bert_hidden_size, t5_hidden_size, max_input_length, 
                 max_output_length, bert_model, t5_model, batch_size):
        super(EncoderDecoder, self).__init__()
        
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.t5_hidden_size = t5_hidden_size
        self.batch_size = batch_size
        
        self.bert = BertModel.from_pretrained(bert_model)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model)
        self.linear = nn.Linear(bert_hidden_size, max_input_length*t5_hidden_size)
    
        self.t5.config.is_encoder_decoder = False
    

    def forward(self, input_ids, input_mask,
                decoder_input_ids, decoder_attention_mask):
        
        # Encode input with BERT
        _, bert_output = self.bert(input_ids=input_ids, attention_mask=input_mask, return_dict=False)
        
        if DEBUG_FLAG: print(f"bert_output - {bert_output.size()}")
        
        # Transform BERT output to T5 input shape
        t5_input = self.linear(bert_output)
        if DEBUG_FLAG: print(f"bert_output linear - {t5_input.size()}")
        if DEBUG_FLAG: print(f"bert_output linear unsqueeze - {t5_input.size()}")
        
        t5_input = t5_input.view(self.batch_size, self.max_input_length, self.t5_hidden_size)
        t5_input = t5_input.unsqueeze(0)
        if DEBUG_FLAG: print(f"t5_input - {t5_input.size()}")
    
        t5_outputs = self.t5(labels=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask,
                             encoder_outputs=t5_input,
                             return_dict = True
                            )
        if DEBUG_FLAG: print(f"t5_input - {type(t5_outputs)}")
        return t5_outputs

    def predict(self, input_ids, input_mask, batch_size, t5_tokenizer):
        
        self.bert.eval()
        _, bert_output = self.bert(input_ids=input_ids, attention_mask=input_mask, return_dict=False)
        if DEBUG_FLAG: print(f"bert_output - {bert_output.size()}")
        
        # Transform BERT output to T5 input shape
        t5_input = self.linear(bert_output)
        if DEBUG_FLAG: print(f"t5_input - {t5_input.size()} - {t5_input}")
        t5_input = t5_input.view(batch_size, self.max_input_length, self.t5_hidden_size)
        t5_input = t5_input.unsqueeze(0)
        if DEBUG_FLAG: print(f"t5_input - {t5_input.size()} - {t5_input}")
        
        # Generate initial input for T5 decoder
        start_token = t5_tokenizer.pad_token_id
        
        decoder_input_ids = torch.tensor([start_token]*batch_size).unsqueeze(0)
        decoder_attention_mask = torch.tensor([1]*batch_size).unsqueeze(0)
        
        if DEBUG_FLAG: print(f"decoder_input_ids - {decoder_input_ids.size()}")
        if DEBUG_FLAG: print(f"decoder_attention_mask - {decoder_attention_mask.size()}")
        
        if DEBUG_FLAG: print(f"initial decoder_input_ids - {decoder_input_ids}")
        
        # Use the model to get output logits
        # Predict the output
        self.t5.eval()
        with torch.no_grad():
            for i in range(127):  # Maximum length of generated sequence
                t5_outputs = self.t5(decoder_input_ids=decoder_input_ids,
                                     decoder_attention_mask=decoder_attention_mask,
                                     encoder_outputs=t5_input)
                if DEBUG_FLAG: print(f"t5_outputs logits - {(t5_outputs.logits).size()}")
    
                next_token_logits = t5_outputs.logits[:, -1, :]
                if DEBUG_FLAG: print(f"next_token_logits - {next_token_logits.size()}")
            
                next_token_id = next_token_logits.argmax(1)
                if DEBUG_FLAG: print(f"next_token_id - {next_token_id.size()}")
                if DEBUG_FLAG: print(f"next_token_id.unsqueeze(-1) - {next_token_id.unsqueeze(-1).size()}")
                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.unsqueeze(-1)], dim=-1)
                decoder_attention_mask = torch.cat([decoder_attention_mask,
                                                    torch.ones_like(next_token_id.unsqueeze(-1))], dim=-1)

                if next_token_id == t5_tokenizer.eos_token_id:
                    break
                
                if DEBUG_FLAG: print(f"pred decoder_input_ids - {decoder_input_ids}")
        
        # generated_text
#         t5_outputs = t5_tokenizer.decode(decoder_input_ids.squeeze(), skip_special_tokens=True)
        t5_outputs = decoder_input_ids #.squeeze()
        
        return t5_outputs

