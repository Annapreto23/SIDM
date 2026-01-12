import sys
import os

log = []
def print_log(msg):
    log.append(str(msg))
    print(msg)

try:
    import colossus
    print_log(f"Colossus file: {colossus.__file__}")
    if hasattr(colossus, '__path__'):
        colossus_path = colossus.__path__[0]
        print_log(f"Colossus path: {colossus_path}")
        try:
            print_log(f"Files in colossus: {os.listdir(colossus_path)}")
        except Exception as e:
            print_log(f"Error listing colossus: {e}")
            
        lss_path = os.path.join(colossus_path, 'lss')
        if os.path.exists(lss_path):
            print_log(f"Files in colossus/lss: {os.listdir(lss_path)}")
        else:
            print_log("colossus/lss does not exist")
    else:
        print_log("colossus has no __path__")

    import colossus.lss
    print_log(f"Imported colossus.lss from {colossus.lss.__file__}")
    print_log(f"Dir(colossus.lss): {dir(colossus.lss)}")

    try:
        from colossus.lss import variance
        print_log(f"Variance module imported: {variance}")
    except ImportError as e:
        print_log(f"Failed to import variance: {e}")

except ImportError as e:
    print_log(f"ImportError: {e}")
except Exception as e:
    print_log(f"Error: {e}")

with open("colossus_debug.log", "w") as f:
    f.write("\n".join(log))
