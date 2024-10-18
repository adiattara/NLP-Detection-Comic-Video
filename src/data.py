import pandas as pd
from sklearn.model_selection import train_test_split

# Let's split the original dataset into training and validation sets and save them as separate CSVs.


def make_dataset(filename):
    return pd.read_csv(filename)

def load_data(path):
    # Load the dataset
    data = pd.read_csv(path)
    return data


def split_data(data):
    # Split the data into train and validation sets
    train_data, validation_data = train_test_split(data, test_size=0.2, random_state=42,stratify=data['is_comic'])
    return train_data, validation_data

def save_split_data(train_data, validation_data, train_data_path, validation_data_path):
    # Save the datasets as CSV files
    train_data.to_csv(train_data_path, index=False)
    validation_data.to_csv(validation_data_path, index=False)
    return train_data_path, validation_data

if __name__ =='__main__' :
    # Load the dataset
    raw_path = '../data/raw/raw.csv'
    data = load_data(raw_path)

    # Split the data
    train_data, validation_data = split_data(data)

    # Save the datasets as CSV files
    train_data_path = '../data/raw/train.csv'
    validation_data_path = '../data/raw/validation.csv'

    save_split_data(train_data, validation_data, train_data_path, validation_data_path)

    # Display paths to confirm saving
    print(train_data_path, validation_data_path)


