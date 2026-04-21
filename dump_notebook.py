import json
import os

notebook_path = '[2026_1]_[Fin]_Difussion_Model_EHM.ipynb'
output_path = 'notebook_dump.txt'

if not os.path.exists(notebook_path):
    print(f"Error: {notebook_path} not found.")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

with open(output_path, 'w', encoding='utf-8') as f:
    for i, cell in enumerate(notebook['cells']):
        cell_type = cell['cell_type'].upper()
        f.write(f"=== CELL {i} ({cell_type}) ===\n")
        if 'source' in cell:
            f.write("".join(cell['source']))
        f.write("\n\n")

print(f"Successfully dumped notebook to {output_path}")
