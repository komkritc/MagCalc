import re
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

ascii_art = r"""
 .----------------.  .----------------.  .-----------------. .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. |
| | ____    ____ | || |     _____    | || | ____  _____  | || |  _________   | |
| ||_   \  /   _|| || |    |_   _|   | || ||_   \|_   _| | || | |  _   _  |  | |
| |  |   \/   |  | || |      | |     | || |  |   \ | |   | || | |_/ | | \_|  | |
| |  | |\  /| |  | || |      | |     | || |  | |\ \| |   | || |     | |      | |
| | _| |_\/_| |_ | || |     _| |_    | || | _| |_\   |_  | || |    _| |_     | |
| ||_____||_____|| || |    |_____|   | || ||_____|\____| | || |   |_____|    | |
| |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------' 
"""

def process_file(filename):
    match = re.match(r't(\d+(\.\d+)?)d(\d+(\.\d+)?)', filename)
    if not match:
        raise ValueError("Invalid filename format")
    
    thickness = float(match.group(1))
    diameter = float(match.group(3))

    df = pd.read_csv(filename, skiprows=6, sep='\t', header=None)
    new_df = pd.DataFrame({
        'thickness': thickness,
        'diameter': diameter,
        'temperature': df.iloc[:, 1],
        'magnetization': df.iloc[:, 6]
    })

    output_filename = f"processed_{filename}.csv"
    new_df.to_csv(output_filename, index=False)
    print(f"Processed file saved as {output_filename}")
    return output_filename

def show_working_path():
    current_path = os.getcwd()
    print(f"Current working directory: {current_path}")
    
def combine_files():
    combined_df = pd.concat([pd.read_csv(f) for f in os.listdir() 
                             if f.startswith("processed_") and f.endswith(".csv")], 
                            ignore_index=True)
    combined_df.to_csv("data4model.csv", index=False)
    print("All processed files combined into data4model.csv")

def list_vampire_files():
    print("Files in the 'vampire' directory:")
    for file in os.listdir():
        print(file)

def delete_processed_files():
    for fname in os.listdir():
        if fname.startswith('processed_'):
            os.remove(fname)
    print("Deleted all files with prefix 'processed_' in the current directory.")

def train_and_convert_model(X_train, X_test, y_train, y_test):
    start_time = time.time()
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    joblib.dump(rf_model, 'rf_model.joblib')
    
    initial_type = [('float_input', FloatTensorType([None, 3]))]
    onx = convert_sklearn(rf_model, initial_types=initial_type, target_opset=12)
    
    with open('rf_model.onnx', 'wb') as f:
        f.write(onx.SerializeToString())
    
    compute_time = time.time() - start_time
    
    return rf_model, compute_time

def compare_predictions(rf_model, X_test):
    sklearn_pred = rf_model.predict(X_test)
    
    sess = rt.InferenceSession("rf_model.onnx", providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    
    X_test_numpy = X_test.to_numpy().astype(np.float32)
    onnx_pred = sess.run([label_name], {input_name: X_test_numpy})[0]
    
    print("Scikit-learn predictions (first 5):", sklearn_pred[:5])
    print("ONNX predictions (first 5):", onnx_pred[:5])
    
    mean_diff = np.mean(np.abs(sklearn_pred - onnx_pred.flatten()))
    print("Mean absolute difference between scikit-learn and ONNX predictions:", mean_diff)

def main():
    print(ascii_art)
    show_working_path()
    vampire_folder = 'vampire'
    delete_processed_files()
        
    if os.path.exists(vampire_folder):
        os.chdir(vampire_folder)
        print(f"Changed working directory to: {os.getcwd()}")
    else:
        print(f"Directory '{vampire_folder}' does not exist. Please create it and add the necessary files.")
        return
    
    list_vampire_files()
    
    for fname in os.listdir():
        if fname.startswith("t"):
            process_file(fname)
    
    combine_files()
    
    print("Loading data...")
    data = pd.read_csv('data4model.csv')
    
    print("Preparing features and target variable...")
    X = data[['thickness', 'diameter', 'temperature']]
    y = data['magnetization']
    
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training the Random Forest Regressor and converting to ONNX...")
    rf_model, compute_time = train_and_convert_model(X_train, X_test, y_train, y_test)
    print(f"Training and conversion took: {compute_time:.2f} seconds")
    
    print("Comparing predictions...")
    compare_predictions(rf_model, X_test)

if __name__ == "__main__":
    main()