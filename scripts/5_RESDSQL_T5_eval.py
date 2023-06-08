#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import torch
from text2sql_decoding_utils import decode_sqls

from tokenizers import AddedToken
from transformers import T5TokenizerFast, T5ForConditionalGeneration
# from transformers.optimization import Adafactor
from transformers.trainer_utils import set_seed

from load_dataset import Text2SQLDataset
from torch.utils.data import DataLoader

from spider_metric.evaluator import EvaluateTool

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


def _test(mode,
          dev_filepath,
          original_dev_filepath,
          save_path,
          db_path,
          batch_size,
          num_beams, num_return_sequences,
          output,
          seed, device):
    
    set_seed(seed)

    start_time = time.time()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    # initialize tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(
        save_path,
        add_prefix_space = True
    )
    
    if isinstance(tokenizer, T5TokenizerFast):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    
    dev_dataset = Text2SQLDataset(
        dir_ = dev_filepath,
        mode = mode
    )

    dev_dataloder = DataLoader(
        dev_dataset, 
        batch_size = batch_size, 
        shuffle = False,
        collate_fn = lambda x: x,
        drop_last = False
    )

    # initialize model
    model = T5ForConditionalGeneration.from_pretrained(save_path)
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    predict_sqls = []
    
    count = 0
    for batch in tqdm(dev_dataloder):
        batch_inputs = [data[0] for data in batch]
        batch_db_ids = [data[1] for data in batch]
        batch_tc_original = [data[2] for data in batch]
        print(batch_inputs)
        print(batch_db_ids)

        tokenized_inputs = tokenizer(
            batch_inputs, 
            return_tensors="pt",
            padding = "max_length",
            max_length = 512,
            truncation = True
        )
        
        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]
        if torch.cuda.is_available():
            encoder_input_ids = encoder_input_ids.cuda()
            encoder_input_attention_mask = encoder_input_attention_mask.cuda()

        # print(encoder_input_ids)
        with torch.no_grad():
            # print("1")
            model_outputs = model.generate(
                input_ids = encoder_input_ids,
                attention_mask = encoder_input_attention_mask,
                max_length = 256,
                decoder_start_token_id = model.config.decoder_start_token_id,
                num_beams = num_beams,
                num_return_sequences = num_return_sequences
            )
            # print("2")
            model_outputs = model_outputs.view(len(batch_inputs), num_return_sequences, model_outputs.shape[1])
            # print("3")
            predict_sqls += decode_sqls(
                db_path, 
                model_outputs, 
                batch_db_ids, 
                batch_inputs, 
                tokenizer, 
                batch_tc_original
            )
            # print("4")
        
        count += 1
        # if count>2: break

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
        print('exact_match score: {}'.format(spider_metric_result["exact_match"]))
        print('exec score: {}'.format(spider_metric_result["exec"]))
    
        return spider_metric_result["exact_match"], spider_metric_result["exec"]


# In[3]:


_test(mode='eval',
      dev_filepath="../data/resdsql_pre/preprocessed_dataset_test.json",
      original_dev_filepath="../data/split/spider_test.json",
      db_path = "/Users/aishwarya/Downloads/spring23/cs685-NLP/project/spider_data/database",
      save_path="models/text2sql/checkpoint-19500",
      batch_size=1,
      num_beams=8,
      num_return_sequences=8,
      output = "models/text2sql/predicted_sql_19500_bs8.txt",
      seed=42,
      device="cpu")


# In[ ]:




