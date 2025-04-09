# Micro-expression Recognition
The complete source code and datasets will be released soon.

# How to run the code

## **Step 1)** Please download the dataset and put it in /datasets
The pre-trained model parameters can be obtained through this link[link](https://drive.google.com/drive/folders/1nkVK2wow5Qo-2lWlaRzU92jzn9Yorlk4?usp=drive_link).


## **Step 2)** Place the files in the structure as follows:
├─datasets----------Training and testing dataset\
├─label_dict-------Label File\
├─utils.py---------Tools\
├─log--------------Save path\
├─model\
├─main.py\
├─test.py\
├─requirements.txt

## **Step 3)** Installation of packages using pip
**pip install -r requirements.txt**

## **Step 4)** Training and Evaluation
**python main.py or test.py**

## **Note for parameter settings**
    --image_folder (Path to the image folder)
    --save_path (Path to save the models)
    --xls_path (Path to Excel file)
    --dataset (Dataset name)
    --cls (Number of classes for classification emotion)
    --load_pretrain (Load pre-trained weights if available)
    --batch_size (Batch size for training and validation)
    --epochs (Number of training epochs)
    --lr (Learning rate)
    --label_path (Label dictionary for expression types)
    --train (Train the model if set, otherwise only evaluate')
#Additional Notes
Please email me at yukejian2021@outlook.com if you have any inquiries or issues.
