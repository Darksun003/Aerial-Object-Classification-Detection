from pathlib import Path

base = Path(r"D:\Internships\labmentix\Aerial Object Classification & Detection\dataset\object_detection_Dataset")

print("Contents of object_detection_Dataset:")
for p in base.iterdir():
    print(" -", p.name)

for sub in ["train", "val", "valid", "test", "images"]:
    p = base / sub
    print(f"Exists {sub}? {p.exists()}")
