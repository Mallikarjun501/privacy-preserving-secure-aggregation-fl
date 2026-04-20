import os
import pandas as pd
import numpy as np
import requests
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader, Subset
from tqdm import tqdm

def download_file(url, local_path):
    if os.path.exists(local_path):
        print(f"{os.path.basename(local_path)} already exists.")
        return
    print(f"Downloading {os.path.basename(local_path)}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_path, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(local_path)
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

def preprocess_nsl_kdd(df):
    df = df.iloc[:, :-1]
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'label'
    ]
    df.columns = columns

    categorical_cols = ['protocol_type', 'service', 'flag']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    numerical_cols.remove('label')
    
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, df.columns.drop('label')

def prepare_data_shards(num_clients=5, data_dir="data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Check if shards are already created
    shards_exist = all(os.path.exists(os.path.join(data_dir, f"shard_{i}.pkl")) for i in range(num_clients))
    if shards_exist and os.path.exists(os.path.join(data_dir, "test_set.pkl")):
        print("Data shards and test set already exist.")
        with open(os.path.join(data_dir, "test_set.pkl"), 'rb') as f:
            test_loader = pickle.load(f)
        with open(os.path.join(data_dir, "num_features.pkl"), 'rb') as f:
            num_features = pickle.load(f)
        return test_loader, num_features

    print("Preparing data shards...")
    train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
    test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
    
    train_path = "KDDTrain+.txt"
    test_path = "KDDTest+.txt"

    download_file(train_url, train_path)
    download_file(test_url, test_path)

    df_train = pd.read_csv(train_path, header=None)
    df_test = pd.read_csv(test_path, header=None)

    df_train, feature_names = preprocess_nsl_kdd(df_train)
    df_test, _ = preprocess_nsl_kdd(df_test)

    X_train = df_train.drop('label', axis=1).values
    y_train = df_train['label'].values
    X_test = df_test.drop('label', axis=1).values
    y_test = df_test['label'].values

    import torch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Save client data shards
    shard_size = len(train_dataset) // num_clients
    for i in range(num_clients):
        start_index = i * shard_size
        end_index = (i + 1) * shard_size if i < num_clients - 1 else len(train_dataset)
        indices = list(range(start_index, end_index))
        client_subset = Subset(train_dataset, indices)
        client_loader = DataLoader(client_subset, batch_size=64, shuffle=True)
        with open(os.path.join(data_dir, f"shard_{i}.pkl"), 'wb') as f:
            pickle.dump(client_loader, f)
        print(f"Saved shard for client {i} with {len(client_subset)} samples.")

    # Save test loader and num_features
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    with open(os.path.join(data_dir, "test_set.pkl"), 'wb') as f:
        pickle.dump(test_loader, f)
    
    num_features = len(feature_names)
    with open(os.path.join(data_dir, "num_features.pkl"), 'wb') as f:
        pickle.dump(num_features, f)

    print("All data shards and test set have been saved.")
    return test_loader, num_features

def load_client_data(client_id, data_dir="data"):
    shard_path = os.path.join(data_dir, f"shard_{client_id}.pkl")
    with open(shard_path, 'rb') as f:
        client_loader = pickle.load(f)
    
    num_features_path = os.path.join(data_dir, "num_features.pkl")
    with open(num_features_path, 'rb') as f:
        num_features = pickle.load(f)
        
    return client_loader, num_features
