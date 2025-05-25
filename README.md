# Stock Market Prediction Using LSTM

This project predicts stock market trends using Long Short-Term Memory (LSTM) neural networks. The model is trained on historical stock price data and deployed using Streamlit.

# Outputs
1.  ![image](https://github.com/user-attachments/assets/a9ab43d3-d27a-4883-8e07-6ce6e1465c5f)
2.  ![image](https://github.com/user-attachments/assets/c9ec8322-2381-4c9a-9833-884c130b3de1)
3.  ![image](https://github.com/user-attachments/assets/d528a6d8-b422-4d93-ae29-db237e74d3c5)
4.  ![image](https://github.com/user-attachments/assets/f243a373-ee3c-41d8-895a-c5a64cb81262)



## Project Structure
- **data/**: Contains raw and preprocessed datasets.
- **models/**: Contains the trained LSTM model in HDF5 format.
- **notebooks/**: Jupyter notebooks for experimentation.
- **app.py**: Main Streamlit application file.
- **requirements.txt**: List of Python dependencies.
- **utils.py**: Helper functions for preprocessing and visualization.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Switch To Virtual Environment
   .\env\Scripts\activate

3. Run the Streamlit Code
   streamlit run end.py


## Directory Tree :
   .
   ├── __pycache__
   ├── data
   │   ├── company_datasets
   │   │   ├── historical_data.csv
   │   │   ├── X.npy
   │   │   └── y.npy
   │   └── env
   │       ├── etc
   │       ├── Include
   │       ├── Lib
   │       ├── Scripts
   │       ├── share
   │       └── pyvenv.cfg
   ├── models
   │   ├── cnn_lstm_model.h5
   │   └── lstm_model.h5
   ├── notebooks
   │   └── env
   ├── .gitignore
   ├── download_data.py
   ├── end.py
   ├── lstm_cnn.py
   ├── lstm.py
   ├── lstm+lstm_with_cnn.py
   ├── randomforest.py
   ├── README.md
   ├── requirements.txt
   └── test.py
   └── utils.py

## Explanation of the Structure:

   # Root Directory :
   Contains all the main files and subdirectories.
   # __pycache__ :
   A hidden directory typically used by Python to store compiled bytecode files.
   # data :
   Contains datasets and related files.
   # company_datasets :
   Subdirectory holding dataset files such as historical_data.csv, X.npy, and y.npy.
   # env :
   A virtual environment setup for managing dependencies. It includes standard directories like etc, Include, Lib, Scripts, share, and the configuration file pyvenv.cfg.
   # models :
   Contains trained model files in .h5 format, specifically:
   # cnn_lstm_model.h5
   lstm_model.h5
   # notebooks :
   Likely contains Jupyter Notebook environments or related configurations.
   Files in the Root Directory :
   # .gitignore : Used to specify files or directories that should be ignored by Git version control.
   # download_data.py : Script for downloading data.
   # end.py : Possibly a script for final processing or cleanup.
   # lstm_cnn.py : Script for LSTM-CNN model-related tasks.
   # lstm.py : Script for LSTM model-related tasks.
   # lstm+lstm_with_cnn.py : Script for combining LSTM and CNN models.
   # randomforest.py : Script for Random Forest model-related tasks.
   # README.md : Markdown file providing project documentation or instructions.
   # requirements.txt : File listing Python dependencies required for the project.
   # test.py : Script for testing purposes.
   # utils.py : Utility functions or helper scripts.