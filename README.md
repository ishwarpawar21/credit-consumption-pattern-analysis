# Consumption Pattern Analysis

This project aims to analyze credit/debit card consumption patterns and predict future consumption using advanced AI algorithms.

## Project Structure

- `data/`: Contains the dataset files.
- `notebooks/`: Jupyter notebooks for data analysis and modeling.
- `scripts/`: Python scripts for data processing and model training.
- `requirements.txt`: List of dependencies.

## Setup

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/consumption-pattern-analysis.git
    cd consumption-pattern-analysis
    ```

2. **Download the dataset:**
    - Download the credit card transactions data from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions?select=User0_credit_card_transactions.csv).
    - Save the file as `credit_card_transactions.csv` in the `data/` directory.

3. **Set up Conda environment:**
    - Create a new Conda environment:
      ```sh
      conda create --name consumption-pattern-analysis python=3.8
      ```
    - Activate the environment:
      ```sh
      conda activate consumption-pattern-analysis
      ```
    - Install dependencies:
      ```sh
      pip install -r requirements.txt
      ```

4. **Run the Jupyter Notebook:**
 
    
     Note: Ensure the Flask server is running and and all pre-requsite Jupyter Notebooks are run from /notbook step by step. 

    - Step 1: Data profiling.
    ```sh
    jupyter notebook notebooks/step_1_data_profiling.ipynb
    ```
    - Step 2: Data sampling
    ```sh
    jupyter notebook notebooks/step_2_data_sampling.ipynb
    ```
    - Step 3: clean and transformation
    ```sh
    jupyter notebook notebooks/step_3_data_cleaning_transformation.ipynb
    ```
    - Step 4: Train, testing and save model 
    ```sh
    jupyter notebook notebooks/step_4_model_training.ipynb
    ```
    - Step 5: Deploy server using saved mdoel
    ```sh
    python app/app.py
    ```
    - Step 6: Access model using API. 
    ```sh
    jupyter notebook consumption-pattern-analysis.ipynb
    ```
## Usage

1. **Load and preprocess the data:**
    - Ensure the dataset is in the `data/` directory and load it in the notebook or scripts.
    - Preprocess the data by handling missing values, encoding categorical variables, etc.

2. **Perform feature engineering:**
    - Extract useful features from the raw data to improve model performance.

3. **Train the model and make predictions:**
    - Use the provided scripts or notebooks to train different models and make predictions on future consumption patterns.

4. **Deploy:**
    - Use the provided scripts or notebooks to  pridict and predictions on next week consumption.   

## Dependencies

The project requires the following Python packages, which are listed in `requirements.txt`:
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow
- statsmodels
- prophet

These packages can be installed via `pip`:
```sh
pip install -r requirements.txt