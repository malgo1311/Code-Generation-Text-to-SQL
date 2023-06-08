# TextToSQL_AdvNLP

Team Members: Aishwarya, Atharva, Dishank, Mrunal

### Install Dependencies

```
pip3 install -r requirements.txt
```

## Contents

1) `data` folder has the preprocessed data
2) `scripts` has all the seq2seq code
3) `scripts/prompt_engineering` has the code for the promting part
4) `reports` folder has the final report

### Baseline model

Our baseline model is a seq2seq model with an encoder-decoder architecture (Iacob et al., 2020). The encoder component employs a bidirectional Long Short-Term Memory (LSTM) network to en- code the input query, while the decoder compo- nent is an LSTM network responsible for generat- ing the SQL query based on the encoded vector.

### Models implemented by us

1) BERT-to-T5 (`scripts/6_BERT+T5+train v3.ipynb`)
2) GPT2-to-T5 (`scripts/8_GPT2+T5+train.ipynb`)

### Experiments

We conducted the following series of experiments:

1) Normal Seq2seq (max input len = 43, max output len = 127)

a) Baseline
b) T5-to-T5 (bs = 32, lr = 3e-5)
c) BERT-to-T5 (bs = 32, lr = 1e-4)
d) GPT2-to-T5 (bs = 32, lr = 1e-4)

2) RESDSQL Seq2seq (max input len = 512, max output len = 256)

a) T5-to-T5 (bs = 16, lr = 3e-4)
b) BERT-to-T5 (bs = 4, lr = 1e-5)
c) GPT2-to-T5 (bs = 4, lr = 1e-5)

3) Schema Classifier (bs = 2, lr = 3e-3)

### Prompting

We noticed that GPT-3.5 was significantly bet- ter on the spider dataset as compared to open- source models (gpt4all-l13b-snoozy, vicuna-13b- 1.1-q4.2). This may be because gpt-3.5 likely has the spider dataset in its training data as compared to other open-source models. On our test data set gpt-3.5 was better but the difference between the open source models was not very significant. We created a new test dataset for this very reason, re- alizing that GPT-3.5 was likely trained on the Spi- der Dataset before. We had an exact match of 78% on the Spider Dataset using GPT3.5 and an exact match of 66% on (gpt4all-l13b-snoozy) and an ex- act match of 64% on (vicuna-13b-1.1-q4.2). On our test set we noticed that gpt3.5 was significantly slower but had a similar exact match of 75% while the two open source models (gpt4all-l13b-snoozy, vicuna-13b-1.1-q4.2) had an exact match of 72% and 70% respectively.

### Conclusion

In this project we focused on the TEXT-to-SQL task. We employed a variety of techniques, including pre-processing, hybrid model architectures and prompt engineering. We explored different models with encoder-decoder architectures, such as T5-to-T5, BERT-to-T5, and GPT-to-T5 and fine- tuned them on the Spider dataset. The models were trained to generate target SQL queries from input natural language text using a seq2seq setup. We improved the accuracy of SQL query genera- tion using Langchain’s zero shot chain of thought prompting method. The model’s performance was evaluated using exact match (EM) and execution accuracy metrics.

**The results showed that the RESDSQL approach significantly improved the performance across all seq2seq models, with the most notable improvement observed in the T5-to- T5 model.**
