import os
import colossus

colossus_path = os.path.dirname(colossus.__file__)

with open("sigma_locations.txt", "w") as out:
    out.write(f"Searching in: {colossus_path}\n")
    for root, dirs, files in os.walk(colossus_path):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if "def sigma" in line:
                                out.write(f"Found in {path} at line {i+1}: {line.strip()}\n")
                except Exception as e:
                    out.write(f"Error reading {path}: {e}\n")
