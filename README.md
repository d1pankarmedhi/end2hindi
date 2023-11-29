# English to Assamese Tarnslation 

A custom model trained on eng to hindi dataset for translation.

## Dataset

For training, I've used [ai4bharat/samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) dataset from Huggingface. With over 140k records of data for Assamese alone, this dataset has in total 11 languages. Such as 

```
languages = ['as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']
```
Below is what the dataset looks like

```
DatasetDict({
    train: Dataset({
        features: ['idx', 'src', 'tgt'],
        num_rows: 141227
    })
})
```
The `src` represents the English source data and likewise, `tgt` represents the Assamese lagnuage. 

Here's a sample data 
```
{
    'idx': 1,
    'src': 'Nevertheless, he gave this assurance: He that has endured to the end is the one that will be saved.',
    'tgt': 'যিয়েই নহওঁক, যিসকলে এই কাৰ্য্য প্ৰাণপণে কৰা চেষ্টা কৰিব, তেওঁলোকক আশ্বাস দি যীচুৱে এইদৰে কৈছিল: “যি জনে শেষলৈকে সহি থাকে, সেই জনেই পৰিত্ৰাণ পাব । ”'
}
```

### Model used

The huggingface model [Helsinki-NLP/opus-tatoeba-en-ro](https://huggingface.co/Helsinki-NLP/opus-tatoeba-en-ro) is used for this fine-tuning task.

