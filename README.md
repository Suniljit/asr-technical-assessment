# ASR Technical Assessment

## Set Up
1. Clone the repository
2. Install the dependencies using `pip install -r requirements.txt`

### Note 
TorchAudio might require ffmpeg to be installed in your device. If you experience any issues, you can install ffmpeg using the following command: `brew install ffmpeg`

## Set Up for Task 5B
In Task 5B, we'll be using an embeddings model by installing the InstructorEmbedding library. This library and its requirements conflict the dependencies for the other tasks. Hence, we will use a seperate enviroment for this task. Below are the instructions:
1. Create a new enviroment with Python 3.10: `conda create --name myenv2 python=3.10`
2. Activate the enviroment: `conda activate myenv2`
3. Install the following packages using conda forge: `conda install -c conda-forge huggingface_hub=0.11.1 sentence-transformers==2.2.2 transformers==4.20.0 InstructorEmbedding pandas scikit-learn`

### Note
If you experience any issues during the installation with the tokenizer package, you will need to ensure that your device has rust installed. Follow the below steps to install it:
1. Run the following command: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. Add just to your environment path: `source $HOME/.cargo/env`
3. Check that rust is installed: `rustc --version`

## Download Dataset
Dataset reference [here](https://www.kaggle.com/datasets/mozillaorg/common-voice)  
Download [here](https://www.dropbox.com/scl/fi/i9yvfqpf7p8uye5o8k1sj/common_voice.zip?rlkey=lz3dtjuhekc3xw4jnoeoqy5yu&dl=0)  

# Task 2
## Objective
Three files, stored in the asr folder, are created in Task 2, to achieve the following objectives:
- `asr_api.py` - Contains the API server that will be used to transcribe audio files.
- `cv-decode.py` - Generates transcribed text from the API and adds a column called 'generated_text' to a csv file
- `DockerFile` - Docker container for the API server

The model we'll be using is the Wav2Vec2 model. Link [here](https://huggingface.co/facebook/wav2vec2-large-960h)

## Using the API
All commands should be run from the home directory, else adjust the paths accordingly. 
### Start The API Server
Run the below command to start the API server  
`python asr/asr_api.py`  
Leave this running in the background to use the API

### Ping Test
Run the below command on a new terminal window. You will receive a 'pong' response if the server is running
```curl -X GET http://localhost:8001/ping```

### Testing the ASR API with Audio Files
Run the below command in a new terminal window. You will receive an output showing the transcribed text and its duration.  
```curl -F 'file=@audio_sample/sample-000000.mp3' http://localhost:8001/asr```

### Transcribing The Dataset
You can run the following code to transcribe the dataset.  
```python asr/cv-decode.py data/common_voice/cv-valid-dev.csv data/common_voice/cv-valid-dev/ asr/transcribed_data.csv```  
Three 3 input arguements are required after calling the cv-decode.py script
1. csv file location 
2. audio folder location
3. output file location to store our transcribed data in a csv

### Build and run the Docker Container
Instead of running the API server locally, you can build and run the docker container. Instructions below:
- ```docker build -t asr-app -f asr/DockerFile .``` to build the docker container 
- ```docker run -p 8001:8001 asr-app``` to run the docker container
- You can use any of the early code to test that the container works as intended. 

# Task 3
## Objective
We will be finetuning a pretrained Wav2Vec2 model on the Common Voice dataset valid-train dataset. As this dataset is large (~200k files), we will be using a subset of the dataset (10k files) to train the model. The dataset will be split into a 70-30 split for training and validation.

The files for Task 3 are stored in the asr-train folder. 

## Training the Model
### Preprocessing Tools and Hyperparameters
Below details the Preprocessing Tools and Hyperparameters used to train the model:
|Tool / Hyperparamter|Description|
|---|---|
|Wav2Vec2Processor|We use the default tokenizer and feature extractor provided by Wav2Vec2 as they are pre-tuned to handle typical speech-to-text tasks effectively. The processor ensures optimal pre-processing of audio and tokenization of text, and is ready for use out of the box.|
|Wav2Vec2ForCTC|The Wav2Vec2ForCTC model is specifically designed for Connectionist Temporal Classification (CTC) tasks, which makes it ideal for end-to-end speech recognition. This model allows us to directly predict text sequences from audio inputs.|
|Learning Rate|1e-5 is used. This is a good starting point for finetuning models.|
|Batch Size|8 is used as a larger amount will trigger an out of memory error.|
|Epochs|30 is selected based on other online examples for finetuning audio modols. Those models use smaller datasets of ~2k files, so we would expect our early stop to be triggered since we are using a significantly larger dataset.|
|Weight Decay|0.01 is applied to regularize the model by penalizing large weights, which helps prevent overfitting. This value is a common starting point in fine-tuning tasks and is effective for improving generalization.|

### Finetuning 
We can follow the code in `asr-train/cv_train_2a.ipynb` to fine tune the model.  
 The model is saved in the `model` folder. 

 ### Results
 - Early stop was triggered after 20 epochs. We can see that after epoch 12, the model had started to overfit. 
- The training loss steadily decreases after each epoch. 
- The validation loss is relatively stable, with some fluctutation between the epochs. Furthermore, the validation loss does not decrease much. This indicates that the model performance on unseen data does not vary significantly.  

#### Chosen Model
As we saved the checkpoints every 5 epochs, we will choose the model at epoch 15, since it only just started to overfit and it still has a low validation loss.

#### Future Work
We are limited in time and compute, if there is future opportunity to train the model again, we could try the following:
- Increase the regularization used (e.g. higher weight decay or introduce dropout)
- Implement learning rate scheduling to optimize the loss convergence
- Try different learning rate values
- Use a larger dataset for training (we are only using 5% of the data available)

### Evaluation 
#### Word Error Rate (WER): 0.0756 (7.56%)
- The WER measure the percentage of words that are incorrectly predicted compared to the ground truth. It accounts for insertions, deletions and substitutions.
- Our WER of 7.56% indicates that, on average, about 7.5 words out of a 100 are predicted incorrectly.This is a good result as it shows that the model performs well on recognizing words in the unseen dataset. 

#### Character Error Rate (CER): 0.0328 (3.28%)
- CER measures the percentage of incorrectly predicted characters compared to the ground truth. It is similar to WER but focuses on character-level errors.
- Our CER of 3.28% indicates that, on average, about 3.3 characters out of a 100 are predicted incorrectly. This is a good result as it shows that the model performs well on recognizing characters in the unseen dataset.

#### Conclusion
- With both a low WER and CER, we can conclude that the fine-tuned model performs well on the unseen test set.


# Task 4
In this task, we will compare the results of a base Wav2Vec2 model vs our finetuned model. 

You can follow the notebook `task-4.ipynb` to process the data for comparison.

### Model Evaluation
|Metric|Base Model|Finetuned Model|Improvement|
|---|---|---|---|
|WER|0.108120|0.078751|0.029369|
|CER|0.045444|0.034017|0.011427|

- The finetuned model outperforms the base model in both WER and CER.
- For further details on the fine-tuned models, please refer to the training-report PDF. 

# Task 5A
The file detected.txt was generated, which contains the filenames that contain the hotwords "be careful", "destroy" and "stranger".

# Task 5B
We used an embeddings model to find files that that similar phrases to the three hot words from Task 5A. The results are as follows:
- 12 of the 15 files in detected.txt were found in the similarity data.
- 3 of the 15 files in detected.txt were not found in the similarity data. But they had a high similarity score, just under the threshold of 0.85.
- 16 extra files, not found in detected.txt, were found in the similarity data. Looking through some of them, they have words that are similar to the hot words, which could be why they were included in the similarity data.

