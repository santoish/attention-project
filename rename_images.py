
from pathlib import Path

folder = Path(input("Enter image folder path: ").strip().strip('"'))
prefix = input("Enter name (e.g. inattentive): ").strip()

extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

if not folder.is_dir():
    print("Folder not found!")
else:
    files = sorted(
        [f for f in folder.iterdir()
         if f.is_file() and f.suffix.lower() in extensions],
        key=lambda f: f.name.lower()
    )

    print(f"Found {len(files)} images.")

    if input("Type YES to rename: ").strip().lower() == "yes":
        # Rename to temporary names first to avoid name conflicts.
        temp_files = []
        for i, file in enumerate(files, start=1):
            temp = folder / f"__temp_rename_{i}__{file.suffix.lower()}"
            while temp.exists():
                temp = folder / f"__temp_rename_{i}_{len(temp_files)}__{file.suffix.lower()}"
            file.rename(temp)
            temp_files.append((temp, file.suffix.lower()))

        for i, (temp, extension) in enumerate(temp_files, start=1):
            new_name = folder / f"{prefix} {i:02d}{extension}"
            temp.rename(new_name)
            print(f"Renamed: {new_name.name}")

        print("All images renamed successfully!")
