# prepare_multilabel_data.py
import os
import pandas as pd
import shutil
from tqdm import tqdm
from config import DATA_DIR

def process_aod(aod_path, out_img_dir):
    """Copy AOD images and assign multi‑hot labels."""
    records = []
    disease_mapping = {
        'amd': 'amd',
        'cataract': 'cataract',
        'diabetes': 'diabetes',
        'glaucoma': 'glaucoma',
        'hypertension': 'hypertension',
        'myopia': 'myopia',
        'normal': 'normal',
        'other': 'other'   # 'other' contains images with no target disease
    }
    base_aod = os.path.join(aod_path, 'collected dataset')
    for folder, label_name in disease_mapping.items():
        folder_path = os.path.join(base_aod, folder)
        if not os.path.isdir(folder_path):
            continue
        print(f"Processing AOD folder: {folder}")
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Create a unique image ID
                img_id = f"aod_{folder}_{fname.split('.')[0]}"
                src = os.path.join(folder_path, fname)
                # Keep the original extension
                ext = os.path.splitext(fname)[1].lower()
                dst = os.path.join(out_img_dir, f"{img_id}{ext}")
                shutil.copy2(src, dst)

                # Multi‑hot labels for our target diseases
                record = {
                    'image_id': img_id,
                    'ckd': 1 if folder == 'diabetes' else 0,       # diabetes as CKD proxy
                    'hypertension': 1 if folder == 'hypertension' else 0,
                    'diabetes': 1 if folder == 'diabetes' else 0,
                    'amd': 1 if folder == 'amd' else 0,
                    'glaucoma': 1 if folder == 'glaucoma' else 0,
                    'cataract': 1 if folder == 'cataract' else 0,
                    'myopia': 1 if folder == 'myopia' else 0,
                    'normal': 1 if folder == 'normal' else 0
                }
                records.append(record)
    return pd.DataFrame(records)

def process_rfmid(rfmid_path, out_img_dir):
    """Process RFMiD training set: copy images and map labels."""
    # Path to the training CSV (nested as we saw)
    train_csv = os.path.join(rfmid_path, 'Training_Set', 'Training_Set', 'RFMiD_Training_Labels.csv')
    if not os.path.exists(train_csv):
        print("RFMiD training CSV not found at:", train_csv)
        return pd.DataFrame()

    df_labels = pd.read_csv(train_csv)
    print("RFMiD CSV columns:", df_labels.columns.tolist())
    print("First few rows:")
    print(df_labels.head())

    # Map our target disease names to actual column names in the RFMiD CSV.
    # Adjust this dictionary based on the printed columns above.
    # Explanation of the mapping:
    #   ckd          -> 'DN' (Diabetic Nephropathy) as a proxy for CKD risk
    #   hypertension -> 'HR' (Hypertensive Retinopathy)
    #   diabetes     -> 'DR' (Diabetic Retinopathy)
    #   amd          -> 'ARMD' (Age‑related Macular Degeneration)
    #   glaucoma     -> (no direct column; leave as None, will be set to 0)
    #   cataract     -> (no direct column; leave as None)
    #   myopia       -> 'MYA' (Myopia)
    #   normal       -> we will compute as 1 only if ALL the above disease columns are 0
    column_mapping = {
        'ckd': 'DN',
        'hypertension': 'HR',
        'diabetes': 'DR',
        'amd': 'ARMD',
        'glaucoma': None,
        'cataract': None,
        'myopia': 'MYA'
    }

    records = []
    # Image directory: Training_Set/Training_Set/Training/
    img_base = os.path.join(rfmid_path, 'Training_Set', 'Training_Set', 'Training')

    for idx, row in tqdm(df_labels.iterrows(), total=len(df_labels), desc="RFMiD"):
        img_id = f"rfmid_{row['ID']}"
        # Try .png first, then .jpg
        img_path = os.path.join(img_base, f"{row['ID']}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_base, f"{row['ID']}.jpg")
        if not os.path.exists(img_path):
            continue

        # Copy image
        ext = os.path.splitext(img_path)[1].lower()
        dst = os.path.join(out_img_dir, f"{img_id}{ext}")
        shutil.copy2(img_path, dst)

        # Build multi‑hot record
        record = {'image_id': img_id}
        disease_present = False  # will become True if any mapped disease is 1
        for target, col in column_mapping.items():
            if col is not None and col in df_labels.columns:
                val = 1 if row[col] == 1 else 0
                record[target] = val
                if val == 1:
                    disease_present = True
            else:
                record[target] = 0
        # Normal = 1 if no disease from our mapped list is present
        record['normal'] = 0 if disease_present else 1
        records.append(record)

    return pd.DataFrame(records)

def main():
    combined_img_dir = os.path.join(DATA_DIR, 'combined', 'images')
    os.makedirs(combined_img_dir, exist_ok=True)

    aod_path = os.path.join(DATA_DIR, 'AOD')
    rfmid_path = os.path.join(DATA_DIR, 'RFMiD')

    print("Processing AOD...")
    aod_df = process_aod(aod_path, combined_img_dir)
    print(f"AOD: {len(aod_df)} images processed.")

    print("\nProcessing RFMiD...")
    rfmid_df = process_rfmid(rfmid_path, combined_img_dir)
    print(f"RFMiD: {len(rfmid_df)} images processed.")

    combined_df = pd.concat([aod_df, rfmid_df], ignore_index=True)
    # Save the combined labels
    out_csv = os.path.join(DATA_DIR, 'combined', 'multilabel.csv')
    combined_df.to_csv(out_csv, index=False)
    print(f"\nTotal combined images: {len(combined_df)}")
    print("Labels saved to", out_csv)
    print("\nFirst 5 rows of the label file:")
    print(combined_df.head())

if __name__ == '__main__':
    main()