import os
import zipfile
import shutil

def unzip_and_clean(target_directory):
    """
    Recursively unzips all .zip files in the target directory and its subdirectories,
    and removes the 'meta' subdirectory from the extracted contents.
    """
    if not os.path.exists(target_directory):
        print(f"Directory not found: {target_directory}")
        return

    # Walk through all directories recursively
    for root, dirs, files in os.walk(target_directory):
        # Check all files in the current directory
        for filename in files:
            file_path = os.path.join(root, filename)

            # Check if it's a zip file
            if filename.lower().endswith('.zip'):
                # Create a directory name based on the zip file (removing .zip extension)
                extract_folder_name = os.path.splitext(filename)[0]
                extract_path = os.path.join(root, extract_folder_name)
                
                # Unzip
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                    
                    # Remove 'meta' directory if it exists
                    meta_dir = os.path.join(extract_path, 'meta')
                    if os.path.exists(meta_dir) and os.path.isdir(meta_dir):
                        shutil.rmtree(meta_dir)
                        
                    os.remove(file_path)
                except zipfile.BadZipFile:
                    print(f"Error: {filename} is a bad zip file.")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Base directory containing all movement data folders
    base_dir = r"C:\Users\roelv\Documents\Machine Learning\AIS\assignment 1"
    
    # Process all movement data directories
    movement_types = ["Door movement data", "letter O movement data", "throwing ball movement data"]
    
    for movement_type in movement_types:
        target_dir = os.path.join(base_dir, movement_type)
        if os.path.exists(target_dir):
            print(f"Processing: {movement_type}")
            unzip_and_clean(target_dir)
    
    print("Done!")
