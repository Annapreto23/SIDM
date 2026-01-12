try:
    with open("verify_final.log", "r", encoding="utf-16") as f:
        print(f.read())
except Exception:
    try:
        with open("verify_final.log", "r", encoding="utf-8") as f:
            print(f.read())
    except Exception as e:
        print(f"Could not read content: {e}")
