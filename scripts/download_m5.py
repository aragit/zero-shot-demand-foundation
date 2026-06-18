import kagglehub
import os
import shutil
import sys

def download_m5():
    print("[Pipeline] Initiating automated M5 competition download...")
    try:
        # competition_download handles the authentication via the ~/.kaggle/access_token we just set
        path = kagglehub.competition_download("m5-forecasting-accuracy")
        
        dest_dir = "data"
        os.makedirs(dest_dir, exist_ok=True)
        
        source_file = os.path.join(path, "sales_train_evaluation.csv")
        dest_file = os.path.join(dest_dir, "sales_train_evaluation.csv")
        
        shutil.copy2(source_file, dest_file)
        print(f"[Pipeline] Successfully deployed M5 evaluation data to: {dest_file}")
    except Exception as e:
        print(f"[Error] Failed to download M5 data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_m5()
