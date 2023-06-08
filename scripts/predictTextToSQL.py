import os

from load_dataset import Text2SQLDataset
from torch.utils.data import DataLoader

from transformers import T5Tokenizer, BertTokenizer

from notebooks.our_scripts.graph import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev_filepath = "../data/resdsql_pre/preprocessed_dataset_test.json"
batch_size = 2
max_input_length = 43
bert_model = 'bert-base-uncased'
t5_model = 't5-small'


# Model class must be defined somewhere
model_path = os.path.join(os.getcwd(), ".\\models\\BERT_T5_lr0.0001_bs2_bert-base-uncased_t5-small\\model_last_99")
#
model2 = torch.load(model_path, map_location=torch.device('cpu'))
print(model2.eval())

# import time
# start_time = time.time()

# initialize tokenizer
bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model)

# if isinstance(tokenizer, T5TokenizerFast):
#     tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

dev_dataset = Text2SQLDataset(
    dir_=dev_filepath,
    mode="train")

dev_dataloder = DataLoader(
    dev_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda x: x,
    drop_last=False
)

# initialize model

model2.eval()
predict_sqls = []
# for batch in tqdm(dev_dataloder):
for idx, batch in enumerate(dev_dataloder):
    batch_inputs = [data[0] for data in batch]
    batch_db_ids = [data[1] for data in batch]
    batch_tc_original = [data[2] for data in batch]

    tokenized_inputs = bert_tokenizer(batch_inputs,
                                      add_special_tokens=True,
                                      padding="max_length",  # True,
                                      max_length=max_input_length,
                                      # pad_to_max_length=True,
                                      return_tensors='pt',
                                      truncation=True)

    encoder_input_ids = tokenized_inputs["input_ids"].to(device)
    encoder_input_attention_mask = tokenized_inputs["attention_mask"].to(device)

    print(f"encoder_input_ids - {encoder_input_ids.size()}")
    print(f"encoder_input_attention_mask - {encoder_input_attention_mask.size()}")

    with torch.no_grad():
        model_outputs = model2.predict(encoder_input_ids, encoder_input_attention_mask,
                                       batch_size=1, t5_tokenizer=t5_tokenizer)

    #         model_outputs = model_outputs.view(len(batch_inputs), opt.num_return_sequences, model_outputs.shape[1])

    #         predict_sqls += decode_sqls(
    #                                     opt.db_path,
    #                                     model_outputs,
    #                                     batch_db_ids,
    #                                     batch_inputs,
    #                                     tokenizer,
    #                                     batch_tc_original
    #                                     )
    break

# new_dir = "/".join(opt.output.split("/")[:-1]).strip()
# if new_dir != "":
#     os.makedirs(new_dir, exist_ok = True)

# # save results
# with open(opt.output, "w", encoding = 'utf-8') as f:
#     for pred in predict_sqls:
#         f.write(pred + "\n")

# end_time = time.time()
# print("Text-to-SQL inference spends {}s.".format(end_time-start_time))

# if opt.mode == "eval":
#     # initialize evaluator
#     evaluator = EvaluateTool()
#     evaluator.register_golds(opt.original_dev_filepath, opt.db_path)
#     spider_metric_result = evaluator.evaluate(predict_sqls)
#     print('exact_match score: {}'.format(spider_metric_result["exact_match"]))
#     print('exec score: {}'.format(spider_metric_result["exec"]))

#     return spider_metric_result["exact_match"], spider_metric_result["exec"]