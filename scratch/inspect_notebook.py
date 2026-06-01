import json
import os

notebook_path = "/Users/lam/Library/CloudStorage/GoogleDrive-he180931@gmail.com/Drive của tôi/AI thực chiến/tts/sommelier/kaggle_notebooks/kaggle incremental/01_v01_diarization.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [22, 26, 27]:
    if idx < len(nb.get('cells', [])):
        cell = nb['cells'][idx]
        print(f"\n=======================================================")
        print(f"CELL {idx} SOURCE:")
        print(f"=======================================================")
        print("".join(cell.get('source', [])))
        print(f"\n=======================================================")
        print(f"CELL {idx} OUTPUTS:")
        print(f"=======================================================")
        for out_idx, out in enumerate(cell.get('outputs', [])):
            out_type = out.get('output_type')
            print(f"  --- Output {out_idx} ({out_type}) ---")
            if out_type == 'stream':
                print("".join(out.get('text', [])))
            elif out_type == 'execute_result':
                data = out.get('data', {})
                if 'text/plain' in data:
                    print("".join(data['text/plain']))
                if 'text/html' in data:
                    print("[HTML output omitted]")
            elif out_type == 'display_data':
                data = out.get('data', {})
                if 'text/plain' in data:
                    print("".join(data['text/plain']))
                if 'text/html' in data:
                    # Let's print some text from the HTML output if it exists
                    html = "".join(data['text/html'])
                    # Print first 200 chars of HTML
                    print(f"[HTML]: {html[:300]}...")
            elif out_type == 'error':
                print("Error: " + out.get('ename', '') + " - " + out.get('evalue', ''))
                print("\n".join(out.get('traceback', [])))
