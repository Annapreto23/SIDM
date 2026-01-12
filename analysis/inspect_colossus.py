import os
import colossus
import inspect

with open("colossus_files.txt", "w") as f:
    f.write(f"Colossus file: {colossus.__file__}\n")
    f.write(f"Dir(colossus): {dir(colossus)}\n")

    try:
        import colossus.lss
        path = os.path.dirname(colossus.lss.__file__)
        f.write(f"LSS Path: {path}\n")
        f.write(f"Files: {os.listdir(path)}\n")
    except Exception as e:
        f.write(f"Error inspecting colossus.lss: {e}\n")
