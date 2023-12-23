# Fine Tuning LLM

Read about fine-tuning language models using the best methods out there. 

![image](https://github.com/d1pankarmedhi/fine-tuning-llm/assets/136924835/d2a1ed0d-5a47-4625-a604-5e64952f2d7c)


- [Fine Tuning LLM](#fine-tuning-llm)
  - [Fine Tuning T5-small for Eng to French Translation](#fine-tuning-t5-small-for-eng-to-french-translation)
    - [Dataset](#dataset)
    - [Model](#model)
    - [HuggingFace repo](#huggingface-repo)
      - [Download and try the model](#download-and-try-the-model)
      - [Training metrics](#training-metrics)


## Fine Tuning T5-small for Eng to French Translation

Fine tuning T5 small 8bit quantized model using LoRA technique using peft and transformers library.

### Dataset 

Dataset used from HuggingFace for fine-tuning,  [**opus100**](https://huggingface.co/datasets/opus100?source=post_page-----287da2d5d7f1--------------------------------), specifically the *fr-en* subset for French and English data.

```python
from datasets import load_dataset
dataset = load_dataset("opus100", "en-fr")
dataset

# output
# DatasetDict({
#     test: Dataset({
#         features: ['translation'],
#         num_rows: 2000
#     })
#     train: Dataset({
#         features: ['translation'],
#         num_rows: 1000000
#     })
#     validation: Dataset({
#         features: ['translation'],
#         num_rows: 2000
#     })
# })
```

### Model

[**T5-small**](https://huggingface.co/t5-small?source=post_page-----287da2d5d7f1--------------------------------) from HuggingFace for translation.

```python
from transformers import AutoModelForSeq2SeqLM
model_id="t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id, 
    load_in_8bit=True, 
    device_map="auto",
)
```

### HuggingFace repo

Checkout the fine-tuned model on [huggingface](https://huggingface.co/dmedhi/eng2french-t5-small) trained on a free T4 google colab GPU.

#### Download and try the model
```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dmedhi/eng2french-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
model = PeftModel.from_pretrained(model, "dmedhi/eng2french-t5-small")

context = tokenizer(["Do you want coffee?"], return_tensors='pt')
output = model.generate(**context)
result = tokenizer.decode(output[0], skip_special_tokens=True)
print(result)

# Output
# Tu veux du café?
```

#### Training metrics

```yaml
# metrics
train_runtime = 1672.4371
train_samples_per_second = 23.917
train_steps_per_second = 2.99
total_flos = 685071170273280.0
train_loss = 1.295289501953125
epoch = 20.0
```