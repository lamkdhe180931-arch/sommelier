import json
import os

def main():
    full_path = "kaggle_notebooks/04_v04_full_mossformer_phowhisper_local_speaker_safe.ipynb"
    out_path = "kaggle_notebooks/kaggle incremental/02_v02_stage_1_original_unmodified.ipynb"
    
    with open(full_path, "r", encoding="utf-8") as f:
        full_nb = json.load(f)
        
    # Find where Stage 2 starts
    stage2_idx = -1
    for i, cell in enumerate(full_nb["cells"]):
        if cell["cell_type"] == "markdown":
            src = "".join(cell["source"])
            if "## 9. Stage 02" in src:
                stage2_idx = i
                break
                
    if stage2_idx == -1:
        print("Lỗi: Không tìm thấy Stage 02 trong notebook gốc!")
        return
        
    # Slice the notebook up to Stage 2
    full_nb["cells"] = full_nb["cells"][:stage2_idx]
    
    # Update the title markdown
    full_nb["cells"][0]["source"][0] = "# Sommelier Kaggle - Original Stage 1 (Unmodified for A/B Testing)\n"
    full_nb["cells"][0]["source"].append("\n**Mục đích:** Chạy độc lập Stage 1 nguyên gốc từ bản V4 để so sánh đối chứng (A/B testing) với bản đã tinh chỉnh tham số.\n")
    
    # Create Markdown cell for Export
    export_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Đóng gói kết quả Diarization\n",
            "\n",
            "Nén thư mục kết quả Diarization thành ZIP để tiện tải về."
        ]
    }
    
    # Create Code cell for Export
    export_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import shutil\n",
            "from pathlib import Path\n",
            "from IPython.display import FileLink, display\n",
            "\n",
            "diar_dir = Path(DIAR_DIR)\n",
            "zip_base = Path(\"/kaggle/working/diarization_results_original\")\n",
            "\n",
            "zip_path = shutil.make_archive(\n",
            "    base_name=str(zip_base),\n",
            "    format=\"zip\",\n",
            "    root_dir=str(diar_dir.parent),\n",
            "    base_dir=diar_dir.name,\n",
            ")\n",
            "\n",
            "print(\"Created:\", zip_path)\n",
            "display(FileLink(zip_path))"
        ]
    }
    
    # Append the export cells
    full_nb["cells"].extend([export_md, export_code])
    
    # Save the new notebook
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_nb, f, indent=1)
        
    print(f"✅ Đã tạo thành công bản Stage 1 nguyên gốc (unmodified) tại: {out_path}")

if __name__ == "__main__":
    main()
