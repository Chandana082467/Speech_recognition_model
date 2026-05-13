import os
import shutil

source_path =r"C:\Users\chand\Downloads\Speech-Emotion-Recognition-main\Speech-Emotion-Recognition-main\dataset" # CHANGE THIS
target_path = "dataset"

os.makedirs(target_path, exist_ok=True)

emotions = ["angry", "happy", "sad", "neutral"]

for e in emotions:
    os.makedirs(os.path.join(target_path, e), exist_ok=True)

for folder in os.listdir(source_path):
    folder_lower = folder.lower()
    src_folder = os.path.join(source_path, folder)

    if not os.path.isdir(src_folder):
        continue

    for emotion in emotions:
        if emotion in folder_lower:
            for file in os.listdir(src_folder):
                if file.endswith(".wav"):
                    src_file = os.path.join(src_folder, file)
                    dst_file = os.path.join(target_path, emotion, file)

                    shutil.copy(src_file, dst_file)

print("DONE sorting dataset!")