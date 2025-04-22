import zipfile, os

Pulmonary_dataset = 'path/to/PulmonaryFibrosis_data'  
Emphysema_dataset = 'path/to/Emphysema_data'

patches_zip = os.path.join(Emphysema_dataset, 'patches.zip')
slices_zip = os.path.join(Emphysema_dataset, 'slices.zip')

patches_extract_path = os.path.join(Emphysema_dataset, 'patches')
slices_extract_path = os.path.join(Emphysema_dataset, 'slices')

os.makedirs(patches_extract_path, exist_ok=True)
os.makedirs(slices_extract_path, exist_ok=True)

with zipfile.ZipFile(patches_zip, 'r') as zip_ref:
    zip_ref.extractall(patches_extract_path)
    print(f"[INFO] Extracted: {patches_zip} → {patches_extract_path}")

with zipfile.ZipFile(slices_zip, 'r') as zip_ref:
    zip_ref.extractall(slices_extract_path)
    print(f"[INFO] Extracted: {slices_zip} → {slices_extract_path}")