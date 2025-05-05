import sys

sys.path.append('/content/DLNLP_25_SN12345678/A')  # Importing functions from Task A
import emotions_train

# import libraries, data and tools
import torch
import sklearn
import warnings
import numpy as np
import pprint as pp
import pandas as pd
from tqdm.auto import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.exceptions import UndefinedMetricWarning
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datasets import DatasetDict, Dataset, Features, ClassLabel, Value
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def plot_labels(data,title):
  title = title
  label_counts = data['labels'].value_counts()
  plt.figure()
  plt.barh(label_counts.index, label_counts.array)
  Label = 'Labels'
  plt.title(Label)

  plt.xlabel(title)
  plt.tight_layout()

  return None

def explore_data(data,large_values): # dataset and a label value that need to be downsampled
  large_values = large_values
  df = data
  plot_labels(df,'Original Label Counts')
  

   # renaming for brevity
  target_count = 200  # Example target count
  #column_to_target = df['labels']

  # extracting rows with the overrepresented value (large_value) and the rest
  large_df = df[df['labels'] == large_values]
  other_df = df[df['labels'] != large_values]

  # downsampling the DataFrame with the overrepresented value
  if len(large_df) > target_count:
    large_downsampled_df = large_df.sample(n=target_count, random_state=42)

  # concatenating the downsampled data with the rest
  df_balanced = pd.concat([other_df, large_downsampled_df], ignore_index=True)

  print("Original DataFrame shape:", df.shape)
  print("Balanced DataFrame shape:", df_balanced.shape)
  print(f"\nValue counts in 'labels' of the balanced DataFrame:") 
  print(df_balanced['labels'].value_counts())
  plot_labels(df_balanced,'Updated Label Counts')
  return df_balanced
  

def resize(data):
  labels_to_downsample = ['offensive_language','neither','hate_speech']

  for label in labels_to_downsample: 
    data = explore_data(data,label)
  return data


# Add a labels column with the string labels
# Read in tweet and labels columns
def read_rearrange(file):
  data = pd.read_csv(file)

  # add a labels column
  data["labels"] = None
  data["text"] = data["tweet"]

  classes = [0,1,2] # 0=hate_speech, 1=offensive_language and 2=neither


  for label in classes:
    for i in range(len(data['class'])):

      if data['class'][i] == 0:
        data.loc[i, 'labels'] = 'hate_speech'
      elif data['class'][i] == 1:
        data.loc[i, 'labels'] = 'offensive_language'
      elif data.loc[i, 'class'] == 2:
        data.loc[i, 'labels'] = 'neither'
  data = data[["text",'labels']]

  data = resize(data)

  return data

def process_emotions(file_path): 


  dataset = read_rearrange(file_path)  # READ IN THE CSV FILES AND REMOVE TEXT DUPLICATES
                                       # EXTRACT THE COLUMNS AND EMOTIONS WE WANT

  ##############  PLOT GRAPHS FOR ENTIRE DATASET  #############
  title1 = 'Hate Speech Dataset'
  emotions_train.plot_data(dataset,title1)
  #########################################################

  train_data, val_data, test_data = emotions_train.split(dataset)  # SPLIT DATA

  ##############  PLOT GRAPHS FOR EACH DATASET SPLIT #############
  title2 = 'train'
  title3 = 'val'
  title4 = 'test'
  emotions_train.plot_data(train_data,title2)
  emotions_train.plot_data(val_data,title3)
  emotions_train.plot_data(test_data,title4)
  #########################################################

  ##### CONVERT STRING LABELS INTO INTEGERS #####
  train_data,unique_labels = emotions_train.label_to_id(train_data)
  val_data,unique_labels = emotions_train.label_to_id(val_data)
  test_data,unique_labels = emotions_train.label_to_id(val_data)

  ###### DEFINE FEATURES #####
  class_names = unique_labels.tolist()  # string labels
  label_feature = ClassLabel(names=class_names)

  features = Features({
    'text': Value('string'),
    'labels': label_feature,
  })
  #### CONVERT PANDAS DATAFRAMES INTO DATASET OBJECTS ###
  train_data = Dataset.from_pandas(train_data[['text', 'labels']], features=features)
  val_data = Dataset.from_pandas(val_data[['text', 'labels']], features=features)
  test_data = Dataset.from_pandas(test_data[['text', 'labels']], features=features)

  ### CONVERT DATASET OBJECTS INTO A DATASETDICT
  dataset = DatasetDict({'train': train_data,
                              'validation': val_data,
                              'test': test_data,})

  #### TOKENIZE DATA AND REMOVE UNTOKENIZED DATA
  dataset = dataset.map(emotions_train.tokenize, batched=True,remove_columns=['text'])
  print(dataset)

  return dataset

def main(file_list,num_labels):
  tokenized_dataset = process_emotions(file_list)
  model,best_model_path = emotions_train.train(tokenized_dataset,num_labels)

  ### obtain test data
  _, _, test_data = emotions_train.load_data(tokenized_dataset)
  
  ### Using the best model path, instantiate a new model
  best_model = best_model_path
  num_labels = num_labels

  best_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
  
  checkpoint = torch.load(best_model_path)
  best_model.load_state_dict(checkpoint)
  
  # Evaluate the new model
  _, accuracy, precision, recall, f1, all_labels, all_preds = emotions_train.evaluate_model(best_model,test_data)
  
  
  # Print the scores
  print(f"Accuracy: {(accuracy):.2f}",'%') # already multiplied by 100 inside eval function
  print(f"Precision: {(precision*100):.2f}",'%') 
  print(f"Recall: {(recall*100):.2f}",'%')
  print(f"F1 score: {(f1*100):.2f}",'%')
  
  # Plot a confusability matrix
  cm = confusion_matrix(all_labels, all_preds)
  print("confusability matrix",cm)

  class_names = ['hate_speech', 'offensive_language', 'neither']  

  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
  disp.plot(cmap=plt.cm.Blues)
  plt.title('Confusion Matrix')
  plt.show()

  

  return None

#main('labeled_data.csv',3)



