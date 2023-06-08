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
print(f"Device detected as - {device}")

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