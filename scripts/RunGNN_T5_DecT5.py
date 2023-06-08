# # MOUNTING GOOGLE DRIVE
# from google.colab import drive
# drive.mount('/content/drive')

# import os
# print(os.getcwd())

# wd = '/content/drive/MyDrive/CS 685/cs685_project/notebooks'
# print(os.listdir(wd))
# os.chdir(wd)
# print(os.getcwd())


# !pip install tokenizers
# !pip install transformers
# !pip install sentencepiece
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

from graph import *
'''
GNN not implemented don't run
lot of basic code is reused
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# FOR PRINTING INTERMEDIATE TORCH SIZES
DEBUG_FLAG = True

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
#         self.linear = nn.Linear(bert_hidden_size, t5_hidden_size)
        self.linear = nn.Linear(bert_hidden_size, max_input_length*t5_hidden_size)

        self.gnn = GNNModel(max_input_length, t5_hidden_size, max_input_length)


    def forward(self, input_ids, input_mask,
                decoder_input_ids, decoder_attention_mask):

        # Encode input with BERT
        _, bert_output = self.bert(input_ids=input_ids, attention_mask=input_mask, return_dict=False)

        if DEBUG_FLAG: print(f"bert_output - {bert_output.size()}")

        # Transform BERT output to T5 input shape
        t5_input = self.linear(bert_output)
        graph_g = create_graph(t5_input)
        if DEBUG_FLAG: print(f"bert_output linear - {t5_input.size()}")

        t5_input = t5_input.view(self.batch_size, self.max_input_length, self.t5_hidden_size)
        t5_input = t5_input.unsqueeze(0)
        if DEBUG_FLAG: print(f"t5_input - {t5_input.size()}")

#         t5_outputs = self.t5(decoder_input_ids=decoder_input_ids,
#                              decoder_attention_mask=decoder_attention_mask,
#                              encoder_outputs=t5_input
#                             )
#         if DEBUG_FLAG: print(f"t5_input logits - {(t5_outputs.logits).size()}")
#         return t5_outputs.logits

        t5_outputs1 = self.t5(labels=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask,
                             encoder_outputs=t5_input,
                             return_dict = True
                            )






        # add gnn embeddigns to t5_output


        # h = t5_outputs['encoder_last_hidden_state'].squeeze(1)
        # graph_g.ndata['h'] = h
        # # graph_g = dgl.remove_self_loop(graph_g)
        # graph_g = dgl.add_self_loop(graph_g)

        # assuming its learning the graph attention by itself !!
        # will try to split this into a different network and train for contrasive loss
        # gnn_enc = self.t5(labels=decoder_input_ids,
        #                      decoder_attention_mask=decoder_attention_mask,
        #                      encoder_outputs=t5_input,
        #                      return_dict = True
        #                     )

        # add the two embeddings together
        # combined_embeddings = t5_outputs1.encoder_last_hidden_state + gnn_enc.encoder_last_hidden_state

        # give gnn + t5 encodings (Graphix layers) to final decoder

        # final_outputs = self.t5.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     encoder_hidden_states=t5_outputs1,
        #     return_dict=True
        # )

        if DEBUG_FLAG: print(f"t5_input - {type(t5_outputs1)}")
        return t5_outputs1

    def predict(self, input_ids, input_mask, batch_size, t5_tokenizer):

        _, bert_output = self.bert(input_ids=input_ids, attention_mask=input_mask, return_dict=False)
        if DEBUG_FLAG: print(f"bert_output - {bert_output.size()}")

        # Transform BERT output to T5 input shape
        t5_input = self.linear(bert_output)
        if DEBUG_FLAG: print(f"t5_input - {t5_input.size()} - {t5_input}")
        t5_input = t5_input.view(batch_size, self.max_input_length, self.t5_hidden_size)
        t5_input = t5_input.unsqueeze(0)
        if DEBUG_FLAG: print(f"t5_input - {t5_input.size()} - {t5_input}")

        t5_input = t5_input.to(torch.LongTensor) #int64)

#         t5_outputs = self.t5.generate(t5_input, max_length = 127)

        #######################################

#         # Generate initial input for T5 decoder
        start_token = t5_tokenizer.pad_token_id

#         decoder_input_ids = torch.tensor([start_token] * batch_size).unsqueeze(0)
#         decoder_attention_mask = torch.tensor([1] * batch_size).unsqueeze(0)

        decoder_input_ids = torch.tensor([start_token]).unsqueeze(0)
        decoder_attention_mask = torch.tensor([1]).unsqueeze(0)

#         decoder_input_ids = decoder_input_ids.view(decoder_input_ids.shape[1],
#                                                    decoder_input_ids.shape[0])
#         decoder_attention_mask = decoder_attention_mask.view(decoder_attention_mask.shape[1],
#                                                              decoder_attention_mask.shape[0])

        print(f"decoder_input_ids - {decoder_input_ids.size()}")
        print(f"decoder_attention_mask - {decoder_attention_mask.size()}")

        print(f"initial decoder_input_ids - {decoder_input_ids}")
        # Use the model to get output logits
        # Predict the output
        with torch.no_grad():
            for i in range(50):  # Maximum length of generated sequence
                t5_outputs = self.t5(decoder_input_ids=decoder_input_ids,
                                     decoder_attention_mask=decoder_attention_mask,
                                     encoder_outputs=t5_input)
#                 print(f"t5_outputs - {t5_outputs}")
                print(f"t5_outputs logits - {(t5_outputs.logits).size()}")

                next_token_logits = t5_outputs.logits[:, -1, :]
                print(f"next_token_logits - {next_token_logits.size()}")

#                 next_token_id = torch.argmax(next_token_logits, dim=-1)
                next_token_id = next_token_logits.argmax(1)
#                 print(f"next_token_id - {next_token_id.size()}")
#                 print(f"next_token_id.unsqueeze(-1) - {next_token_id.unsqueeze(-1).size()}")
                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.unsqueeze(-1)], dim=-1)
                decoder_attention_mask = torch.cat([decoder_attention_mask,
                                                    torch.ones_like(next_token_id.unsqueeze(-1))], dim=-1)

                if next_token_id == t5_tokenizer.eos_token_id:
                    break

                print(f"pred decoder_input_ids - {decoder_input_ids}")

#                 break

        # generated_text
        t5_outputs = t5_tokenizer.decode(decoder_input_ids.squeeze(), skip_special_tokens=True)
        #######################################

        return t5_outputs


# must remain same!!
def train(train_filepath, batch_size, bert_hidden_size, t5_hidden_size, lr, num_epochs,
         max_input_length, max_output_length, bert_model, t5_model):

    sub_folder_name = f"BERT_T5_lr{lr}_bs{batch_size}_{bert_model}_{t5_model}"
    models_directory = f"models/{sub_folder_name}"

    if not os.path.isdir(models_directory):
        os.makedirs(models_directory)

    # TENSORBOARD
    writer = SummaryWriter(f'tb/loss_plot/{sub_folder_name}')

    train_dataset = Text2SQLDataset(
            dir_ = train_filepath,
            mode = "train")

    train_dataloder = DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True,
            collate_fn = lambda x: x,
            drop_last = True
        )

    print(f"Number of batches - {len(train_dataloder)}")

    # Define BERT and T5 tokenizers
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model)
    print(f"Tokenizers loaded")

    model = EncoderDecoder(bert_hidden_size, t5_hidden_size,
                           max_input_length, max_output_length,
                           bert_model, t5_model, batch_size).to(device)
    print(f"Model loaded")
#     print(f"{model.config.decoder_start_token_id}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Otimizer - Adam")

    criterion = nn.CrossEntropyLoss(ignore_index=t5_tokenizer.pad_token_id)
    print(f"CrossEntropyLoss initialized")

    # initialize array of losses
    losses = {'train': {}, "val": {}}

    # for epoch in range(num_epochs):
    with trange(num_epochs) as tr:
        for epoch in tr:

            # Train the model
            model.train()

            batch_loss = 0

            for idx, batch in enumerate(train_dataloder):

                batch_inputs = [data[0] for data in batch]
                batch_sqls = [data[1] for data in batch]

                batch_db_schemas = [data[4] for data in batch]
                batch_sql_skeleton = [data[5] for data in batch]
                batch_decoder_input = [data[6] for data in batch]

                if DEBUG_FLAG:
                    if epoch == 0 and idx == 0:
                        print(f"batch_inputs - {type(batch_inputs)} {len(batch_inputs)}")
                        print(f"batch_sqls - {type(batch_sqls)} {len(batch_sqls)}")

#                 for temp_i, temp in enumerate(batch_inputs):
#                     print(f"batch_inputs - {batch_inputs[temp_i]}")
#                     print(f"batch_sqls - {batch_sqls[temp_i]}")

                tokenized_inputs = bert_tokenizer(batch_inputs,
                                                  add_special_tokens=True,
                                                  padding="max_length", #True,
                                                  max_length=max_input_length,
                                                  #pad_to_max_length=True,
                                                  return_tensors='pt',
                                                  truncation=True)

                encoder_input_ids = tokenized_inputs["input_ids"].to(device)
                encoder_input_attention_mask = tokenized_inputs["attention_mask"].to(device)

#                 print(f"encoder_input_ids - {encoder_input_ids}")
                tokenized_outputs = t5_tokenizer(batch_sqls,
                                                 add_special_tokens=True,
                                                 padding="max_length", #True,
                                                 max_length=max_output_length,
                                                 #pad_to_max_length=True,
                                                 return_tensors='pt',
                                                 truncation=True)


                decoder_input_ids = tokenized_outputs["input_ids"].to(device)
                # replace padding token id's of the labels by -100 so it's ignored by the loss
                decoder_input_ids[decoder_input_ids == t5_tokenizer.pad_token_id] = -100
                decoder_attention_mask = tokenized_outputs["attention_mask"].to(device)
#                 labels = None #tokenized_outputs["attention_mask"].to(device)

#                 print(f"decoder_input_ids - {decoder_input_ids}")

                if DEBUG_FLAG and epoch == 0 and idx == 0:
                    print(f"encoder_input_ids - {encoder_input_ids.size()}")
                    print(f"encoder_input_attention_mask - {encoder_input_attention_mask.size()}")
                    print(f"decoder_input_ids - {decoder_input_ids.size()}")
                    print(f"decoder_attention_mask - {decoder_attention_mask.size()}")

                # Clear gradients
                optimizer.zero_grad()

                # # graph
                # g  = create_graph(encoder_input_ids)

                model_output = model(encoder_input_ids,
                               encoder_input_attention_mask,
                               decoder_input_ids,
                               decoder_attention_mask)
#                                labels=labels)

                output = model_output["logits"]
#                 print(f"output - {output.size()}")
#                 print(f"decoder_input_ids - {decoder_input_ids.size()}")

                output_resize = output.view(output.shape[0]*output.shape[1], output.shape[2])
                decoder_input_ids_resize = decoder_input_ids.view(decoder_input_ids.shape[0]*decoder_input_ids.shape[1])

#                 print(f"output_resize - {output_resize.size()}")
#                 print(f"decoder_input_ids_resize - {decoder_input_ids_resize.size()}")

#                 loss = criterion(output_resize, decoder_input_ids_resize)
#                 batch_loss += loss.item()

#                 print(f"output - {model_output}")
                loss = model_output["loss"]
                batch_loss += loss

#                 predicted_classes = torch.argmax(output_resize, dim=-1)

#                 print(f"output_resize - {predicted_classes.size} - {predicted_classes}")
#                 print(f"decoder_input_ids_resize - {decoder_input_ids_resize}")

                # backpropagation
                loss.backward()
                optimizer.step()

                break

            batch_loss /= len(train_dataloder)
            losses['train'][epoch] = f"{batch_loss:.3f}"
            #progress bar
            tr.set_postfix({"epoch_num":epoch,
                            "loss":f"{batch_loss:.10f}"})

            with open(os.path.join(models_directory, "loss.json"), 'w') as f:
                json.dump(losses, f)

            writer.add_scalar('Training loss', batch_loss, global_step=epoch+1)
            # save models
            # if (epoch > 3 and epoch % 5 == 0):
            torch.save(model, os.path.join(models_directory, f"model_{epoch}"))
    torch.save(model, os.path.join(models_directory, f"model_last_{epoch}"))
    print(f"saved: {models_directory} model_last_{epoch}")


# Define hyperparameters

train(train_filepath ="../../data/resdsql_pre/preprocessed_dataset_train.json",
      batch_size = 10,  #32
      bert_hidden_size = 768,
      t5_hidden_size = 512,
      lr = 1e-4,
      num_epochs = 100,  #300
      max_input_length = 512,
      max_output_length = 512,
      bert_model = 'bert-base-uncased',
      t5_model = 't5-small')



