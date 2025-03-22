#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def organize_files(directory):
    # Get absolute path
    directory_path = Path(directory).resolve()
    print(f"Organizing files in: {directory_path}")
    
    # Get all files in the directory
    files = [f for f in directory_path.iterdir() if f.is_file() and f.name != os.path.basename(__file__)]
    
    # Define file type categories based on extensions
    categories = {
        "TextFiles": [".txt"],
        "Documents": [".pdf", ".doc", ".docx"],
        "Scripts": [".sh", ".py"],
        "Images": [".jpg", ".jpeg", ".png", ".gif"]
    }
    
    # Create folders if they don't exist
    for folder in categories.keys():
        folder_path = directory_path / folder
        if not folder_path.exists():
            folder_path.mkdir()
            print(f"Created folder: {folder}")
    
    # Move files to appropriate folders
    for file in files:
        file_ext = file.suffix.lower()
        moved = False
        
        for folder, extensions in categories.items():
            if file_ext in extensions:
                destination = directory_path / folder / file.name
                shutil.move(str(file), str(destination))
                print(f"Moved {file.name} to {folder}/")
                moved = True
                break
        
        if not moved:
            # Create a Misc folder if there are files with unrecognized extensions
            misc_folder = directory_path / "Misc"
            if not misc_folder.exists():
                misc_folder.mkdir()
                print(f"Created folder: Misc")
            
            destination = misc_folder / file.name
            shutil.move(str(file), str(destination))
            print(f"Moved {file.name} to Misc/")
    
    print("File organization complete!")

if __name__ == "__main__":
    # Use the current directory
    organize_files(".") 