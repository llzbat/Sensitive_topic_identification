import pandas as pd
import torch
from Sensitive_Word.word_filter import DFAFilter
from Utils import (
    bert_CNN_Config, bert_CNN_Model,
    bert_Config, bert_Model,
    TextCNN_Config, TextCNN_Model,
    DPCNN_Config, DPCNN_Model,
    FastText_Config, FastText_Model,
    TextRCNN_Config, TextRCNN_Model
)

key = {0: '正常',
       1: '政治',
       2: '违法',
       3: '色情',
       4: '暴恐',
       5: '广告',
       6: '不确定'
       }

torch.backends.cudnn.enabled = False


# Define model configurations and load models
def load_models(config_class, model_class, *args, **kwargs):
    if config_class in [TextCNN_Config, DPCNN_Config, TextRCNN_Config]:
        config = config_class('dataset', 'embedding_SougouNews.npz')
    elif config_class == FastText_Config:
        config = config_class('dataset', 'random')
    else:
        config = config_class('dataset')
    model = model_class(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path, map_location=config.device))
    model.eval()
    return config, model


# Load all models
model_classes = [
     (bert_CNN_Config, bert_CNN_Model),
     (bert_Config, bert_Model),

    (TextCNN_Config, TextCNN_Model),
    (DPCNN_Config, DPCNN_Model),
    (TextRCNN_Config, TextRCNN_Model),
    (FastText_Config, FastText_Model)
]
models_and_configs = [load_models(*classes) for classes in model_classes]


# Sensitive word filtering
def word_filter(text):
    dfa_filter = DFAFilter()
    filtered_content, black_words = dfa_filter.filter_sensitive_words(text)
    return bool(black_words), black_words, filtered_content


# Model prediction
def prediction_model(text, models_and_configs):
    predictions = []
    for config, model in models_and_configs:
        data = config.build_dataset(text)
        with torch.no_grad():
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            score, predict = torch.max(probabilities, dim=1)
            if score.item() < 0.5:
                predict = 6
            predictions.append(int(predict))
    votes = {i: predictions.count(i) for i in set(predictions)}
    final_prediction = max(votes, key=votes.get) if max(votes.values()) >= 3 else 6
    return final_prediction


# Process CSV file
def process_csv(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path, header=None,skiprows=range(1, 200001), nrows=800000)
    df[2] = False  # is_sensitive
    df[3] = None  # sensitivity_category
    valid_rows_mask = [True] * len(df)  # Create a mask to keep track of valid rows

    cnt = 0
    for index, row in df.iterrows():
        cnt += 1
        if cnt % 10000 == 0:
            print(cnt)
        try:
            text = row[0]
            is_sensitive, _, _ = word_filter(text)
            sensitivity_category = prediction_model(text, models_and_configs)
            df.at[index, 2] = is_sensitive
            df.at[index, 3] = key[sensitivity_category]
        except Exception as e:
            print(f"An error occurred while processing the sentence: {index}")
            valid_rows_mask[index] = False  # Mark this row as invalid

    # Use the mask to filter out invalid rows
    df = df[valid_rows_mask]
    df.to_csv(output_csv_path, index=False, header=False)


if __name__ == '__main__':
    input_csv_path = 'merged2.csv'
    output_csv_path = 'output2.csv'
    process_csv(input_csv_path, output_csv_path)
