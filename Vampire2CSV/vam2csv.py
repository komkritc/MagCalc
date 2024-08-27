import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import glob

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


def show_working_path():
    current_path = os.getcwd()
    print(f"Current working directory: {current_path}")

def display_files_with_prefix_t():
    current_path = os.getcwd()
    # Use both glob and os.listdir to check for files
    glob_files = glob.glob('t*')
    listdir_files = [f for f in os.listdir(current_path) if f.lower().startswith('t')]
    
    # Combine and remove duplicates
    all_files = list(set(glob_files + listdir_files))
    
    #if all_files:
    #    print("Files with prefix 't' in the current directory:")
    #    for file in sorted(all_files):
    #        print(f"  - {file}")
    #else:
    #    print("No files with prefix 't' found in the current directory.")
    
    # Debug information
    print("Vampire Data File(s) Information:")
    #print(f"Files found by glob: {glob_files}")
    print(f"Files found: {listdir_files}")
    #print(f"Combined unique files: {all_files}")
    
    return all_files

def process_file(filename):
    # Extract thickness and diameter from filename
    match = re.match(r't(\d+(\.\d+)?)d(\d+(\.\d+)?)', filename)
    if match:
        thickness = float(match.group(1))
        diameter = float(match.group(3))
    else:
        raise ValueError(f"Invalid filename format: {filename}")

    # Read the CSV file, skipping the first 6 rows, using tab as separator, and no header
    df = pd.read_csv(filename, skiprows=6, sep='\t', header=None)

    # Add new columns for thickness and diameter
    df['thickness'] = thickness
    df['diameter'] = diameter

    # Add temperature and magnetization columns
    df['temperature'] = df.iloc[:, 1]  # Get data from the second column
    df['magnetization'] = df.iloc[:, 6]  # Get data from the seventh column

    # Create a new DataFrame with only the desired columns
    new_df = df[['thickness', 'diameter', 'temperature', 'magnetization']]

    # Save the modified dataframe back to a new CSV file
    #output_filename = f"processed_t{thickness}d{diameter}_{os.path.basename(filename)}"
    output_filename = f"processed_{os.path.basename(filename)}"+".csv"
    full_output_path = os.path.join(os.getcwd(), output_filename)
    new_df.to_csv(full_output_path, index=False)

    print(f"Processed file saved as: {full_output_path}")
    return full_output_path

def plot_graph(processed_file):
    # Read the processed CSV file
    df = pd.read_csv(processed_file)

    # Get thickness and diameter values
    thickness = df['thickness'].iloc[0]
    diameter = df['diameter'].iloc[0]

    # Create a scatter plot using dot plot style
    plt.figure(figsize=(10, 8))
    plt.plot(df['temperature'], df['magnetization'], 'o', markersize=5)
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('Magnetization (emu)', fontsize=12)
    plt.title(f'Magnetization vs Temperature\nThickness: {thickness} nm, Diameter: {diameter} nm', fontsize=14)
    plt.grid(True)

    # Add text box with thickness and diameter information
    info_text = f'Thickness: {thickness} nm\nDiameter: {diameter} nm'
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Improve layout
    plt.tight_layout()

    # Save the plot as an image file
    #plot_filename = f"plot_t{thickness}d{diameter}_{os.path.splitext(os.path.basename(processed_file))[0]}.png"
    plot_filename = f"plot_{os.path.splitext(os.path.basename(processed_file))[0]}.png"
   
    full_plot_path = os.path.join(os.getcwd(), plot_filename)
    plt.savefig(full_plot_path, dpi=300)
    plt.close()

    print(f"Graph plotted and saved as: {full_plot_path}")

def process_single_file():
    show_working_path()
    files = display_files_with_prefix_t()
    if not files:
        print("No files to process. Please check the current directory.")
        return
    
    filename = input('Please enter the filename from the list above: ')
    if filename not in files:
        print(f"Error: '{filename}' is not in the list of files with prefix 't'.")
        return
    
    try:
        processed_file = process_file(filename)
        plot_graph(processed_file)
        print(f"Successfully processed file. Output: {processed_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def process_bulk_files():
    show_working_path()
    files = display_files_with_prefix_t()
    if not files:
        print("No files to process. Please check the current directory.")
        return

    proceed = input("Do you want to proceed with processing these files? (y/n): ").lower()
    if proceed != 'y':
        print("Bulk processing cancelled.")
        return

    for filename in files:
        try:
            processed_file = process_file(filename)
            plot_graph(processed_file)
            print(f"Successfully processed file. Output: {processed_file}")
        except Exception as e:
            print(f"An error occurred processing {filename}: {str(e)}")

def main_menu():
    print(ascii_art)
    show_working_path()
    display_files_with_prefix_t()
    while True:
        print("\nMain Menu:")
        print("1. Process single file")
        print("2. Process all files (bulk mode)")
        print("3. Display Vampire files")
        print("4. Exit program")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            process_single_file()
        elif choice == '2':
            process_bulk_files()
        elif choice == '3':
            display_files_with_prefix_t()
        elif choice == '4':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

# Main execution
if __name__ == "__main__":
    main_menu()