# NLP-Multi-Label-Classifier
This project classifies BBC news headlines into categories (sport, business, politics, tech, entertainment) using an LSTM model in TensorFlow. It preprocesses text by tokenizing, removing stopwords, and padding sequences. The model is trained on labeled data and tracks accuracy and loss across epochs for performance evaluation.


# Dependencies 
This project requires the following dependencies:

Python 3.x
TensorFlow 2.x
Keras
NumPy
NLTK

# Method 

1. **Data Preparation**:
   - **Loading Data**: The project reads a CSV file containing BBC news headlines and their corresponding categories.
   - **Text Preprocessing**: Headlines are cleaned by removing stopwords and unnecessary whitespace, which helps improve model performance.

2. **Tokenization**:
   - The text data is tokenized using TensorFlow’s `Tokenizer`, which converts the words into numerical sequences. This allows the model to process the text as numerical data.
   - A vocabulary size of 5000 is set, meaning the model will consider the 5000 most frequent words.

3. **Padding**:
   - To ensure all input sequences have the same length, the tokenized sequences are padded to a maximum length of 200 using `pad_sequences`. This step is essential for training neural networks.

4. **Model Architecture**:
   - The model is built using a sequential architecture that includes:
     - **Embedding Layer**: Converts word indices into dense vectors of fixed size (64 dimensions).
     - **Bidirectional LSTM Layer**: This layer processes the input sequence in both forward and backward directions, capturing dependencies in the text effectively.
     - **Dense Layers**: The output layer uses softmax activation to predict one of the five categories based on the extracted features.

5. **Training**:
   - The model is compiled with a loss function (sparse categorical cross-entropy) and an optimizer (Adam). 
   - It is trained on 80% of the dataset, while the remaining 20% is used for validation to assess model performance.

6. **Evaluation and Prediction**:
   - After training, the model’s accuracy and loss are plotted over epochs to visualize performance.
   - The model can then make predictions on new headlines, providing the predicted category along with the original headline for comparison.

7. **Testing Predictions**:
   - A function is provided to randomly test predictions on new articles, allowing users to see how well the model generalizes to unseen data.


# Results
The model achieves an accuracy of 93.05% on the test set. The confusion matrix and classification report can be found in the Jupyter Notebook.
