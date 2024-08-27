import onnxruntime as ort
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import sys

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

def check_model_file():
    current_directory = os.getcwd()
    model_path = os.path.join(current_directory, "small_rf_model.onnx")
    if not os.path.exists(model_path):        
        print("Please ensure the model file is in the same directory as this script.")
        sys.exit(1)
    return model_path

def load_model(model_path):
    return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    #return ort.InferenceSession(model_path)

def predict_magnetization(session, thickness, diameter, temperature):
    input_data = np.array([[thickness, diameter, temperature]], dtype=np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: input_data})
    return result[0][0][0]

def single_mode(session):
    thickness = float(input("Enter thickness: "))
    diameter = float(input("Enter diameter: "))
    temperature = float(input("Enter temperature: "))
    magnetization = predict_magnetization(session, thickness, diameter, temperature)
    print(f"Predicted magnetization: {magnetization:.6f}")

def auto_mode(session):
    thickness = float(input("Enter thickness: "))
    diameter = float(input("Enter diameter: "))
    
    temperatures = np.arange(0, 901, 2)
    magnetizations = []
    
    for temp in temperatures:
        mag = predict_magnetization(session, thickness, diameter, temp)
        magnetizations.append(mag)
    
    # Save data to CSV
    filename = f"magnetization_data_t{thickness}_d{diameter}.csv"
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Temperature", "Magnetization"])
        writer.writerows(zip(temperatures, magnetizations))
    print(f"Data saved to {filename}")
    
    # Plot graph using dot plot style
    plt.figure(figsize=(10, 6))
    plt.scatter(temperatures, magnetizations, color='blue', marker='o', s=10)
    plt.title(f"Magnetization vs Temperature\nThickness: {thickness}, Diameter: {diameter}")
    plt.xlabel("Temperature")
    plt.ylabel("Magnetization")
    plt.grid(True)
    plt.savefig(f"magnetization_plot_t{thickness}_d{diameter}.png")
    plt.show()

def main_menu():
    print("Magnetic Information Storage Technology, MSU & UoY")
    print(ascii_art)
    print("Checking for ONNX model file...")
    model_path = check_model_file()
    print("Loading ONNX model. Please wait...")
    session = load_model(model_path)
    
    while True:
        print("\nMenu:")
        print("(1) Run single mode")
        print("(2) Run auto mode, Temperature 0 to 900")
        print("(3) Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            single_mode(session)
        elif choice == '2':
            auto_mode(session)
        elif choice == '3':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()