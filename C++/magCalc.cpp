#include <fstream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <iostream>

#include <locale>
#include <codecvt>

// Function to convert std::wstring to std::string
std::string wstring_to_string(const std::wstring& wstr) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(wstr);
}

// Custom function to check if a file exists
bool fileExists(const std::string& filePath) {
    std::ifstream file(filePath);
    return file.good();
}

// Function to display ASCII art
void displayAsciiArt() {
    std::cout << R"(
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
    )" << std::endl;
}

// Function to find a single output value from three inputs
float findSingleValue(float thickness, float diameter, float temperature, Ort::Session& session, const std::vector<const char*>& input_names, const std::vector<const char*>& output_names) {
    // Prepare input tensor
    std::vector<float> input_tensor_values = { thickness, diameter, temperature };
    std::vector<int64_t> input_shape = { 1, 3 }; // Batch size of 1, 3 input features
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // Run inference
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());

    // Get output data
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    return output_data[0];
}

// Function to generate data and save to CSV
void generateDataAndSaveToCSV(float thickness, float diameter, Ort::Session& session, const std::vector<const char*>& input_names, const std::vector<const char*>& output_names) {
    std::ofstream csvFile;
    std::string filename = std::to_string(thickness) + "_" + std::to_string(diameter) + ".csv";
    csvFile.open(filename);

    // Write CSV header
    csvFile << "Temperature,Magnetization\n";

    // Prepare input tensor shape
    std::vector<int64_t> input_shape = { 1, 3 }; // Batch size of 1, 3 input features
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    //for (float temperature = 0; temperature <= 900; ++temperature) {
    for (float temperature = 0; temperature <= 900; temperature += 2) {
        // Prepare input tensor values
        std::vector<float> input_tensor_values = { thickness, diameter, temperature };
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

        // Run inference
        auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());

        // Get output data
        float* output_data = output_tensors[0].GetTensorMutableData<float>();

        // Write data to CSV
        csvFile << temperature << "," << output_data[0] << "\n";
    }

    csvFile.close();
    std::cout << "Data saved to " << filename << std::endl;
}

int main() {

    try {
        std::cout << "\nMagnetic Information Storage Technology, MSU & UoY" << std::endl;
        displayAsciiArt();
        std::cout << "\nMagCalc (MINT Lab) ML version trained from Vampire Data" << std::endl;

        // Initialize ONNX Runtime environment
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
        Ort::SessionOptions sessionOptions;

        // Load the ONNX model
        const std::wstring primaryModelPath = L"rf_model.onnx";
        const std::wstring secondaryModelPath = L"C:\\Users\\komkritc\\OneDrive\\Documents\\magCalc\\c++\\rf_model.onnx";

        std::wstring modelPath;
        // Convert wstring paths to string
        std::string primaryModelPathStr = wstring_to_string(primaryModelPath);
        std::string secondaryModelPathStr = wstring_to_string(secondaryModelPath);

        // Check if the primary model path exists
        if (fileExists(primaryModelPathStr)) {
            modelPath = primaryModelPath;
            std::wcout << L"Using primary model path: " << modelPath << std::endl;
        }
        else if (fileExists(secondaryModelPathStr)) {
            modelPath = secondaryModelPath;
            std::wcout << L"\nPrimary model not found. \nUsing secondary model path: " << modelPath << std::endl;
        }
        else {
            std::cerr << "Neither model path is accessible." << std::endl;
            return -1; // Exit if neither path is valid
        }

        // Load the model 
        std::cout << "\nLoading model, please wait..." << std::endl;
        Ort::Session session(env, modelPath.c_str(), sessionOptions);
        std::cout << "Model loaded successfully." << std::endl;

        // Get input and output names
        Ort::AllocatorWithDefaultOptions allocator;
        size_t numInputNodes = session.GetInputCount();
        std::vector<const char*> input_names(numInputNodes);
        for (size_t i = 0; i < numInputNodes; i++) {
            auto input_name = session.GetInputNameAllocated(i, allocator);
            input_names[i] = input_name.get();
        }

        size_t numOutputNodes = session.GetOutputCount();
        std::vector<const char*> output_names(numOutputNodes);
        for (size_t i = 0; i < numOutputNodes; i++) {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            output_names[i] = output_name.get();
        }

        std::string userInput;
        while (true) {
            std::cout << "\nSelect mode: \n(a) Autorun with temperature from 0 to 900 \n(s) Single value prediction \n(exit) to close:\n> ";
            std::cin >> userInput;

            if (userInput == "exit") {
                break;
            }
            else if (userInput == "a" || userInput == "A") {
                // User inputs for model parameters
                float thickness, diameter;
                std::cout << "Enter thickness: ";
                std::cin >> thickness;
                std::cout << "Enter diameter: ";
                std::cin >> diameter;

                // Generate data and save to CSV
                generateDataAndSaveToCSV(thickness, diameter, session, input_names, output_names);
            }
            else if (userInput == "s" || userInput == "S") {
                // User inputs for model parameters
                float thickness, diameter, temperature;
                std::cout << "Enter thickness: ";
                std::cin >> thickness;
                std::cout << "Enter diameter: ";
                std::cin >> diameter;
                std::cout << "Enter temperature: ";
                std::cin >> temperature;

                // Find single value using the function
                float result = findSingleValue(thickness, diameter, temperature, session, input_names, output_names);
                std::cout << "Magnetization: " << result << std::endl;
            }
            else {
                std::cout << "Invalid input. Please try again." << std::endl;
            }
        }

    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}