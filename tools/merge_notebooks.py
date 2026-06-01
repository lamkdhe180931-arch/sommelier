import json

def main():
    full_path = "kaggle_notebooks/04_v04_full_mossformer_phowhisper_local_speaker_safe.ipynb"
    s1_path = "kaggle_notebooks/kaggle incremental/01_v01_diarization.ipynb"
    out_path = "kaggle_notebooks/kaggle incremental/00_v00_full_pipeline_with_updated_stage_1.ipynb"
    
    with open(full_path, "r", encoding="utf-8") as f:
        full_nb = json.load(f)
        
    with open(s1_path, "r", encoding="utf-8") as f:
        s1_nb = json.load(f)
        
    # Lấy TẤT CẢ các cell trong Stage 1 của 01_v01_diarization (từ header "## 7. Stage 01 - Speaker diarization" đến cuối hoặc trước phần khác)
    # Tuy nhiên, người dùng thường quan tâm đến cell gọi lệnh chạy stage 1 (có tham số mới). 
    # Thay thế gọn gàng cell chứa lệnh chạy `stage_01_diarize.py` là an toàn nhất để không làm hỏng flow Full Pipeline.
    
    s1_code = None
    for cell in s1_nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "stage_01_diarize.py" in src and "run_logged" in src:
                s1_code = cell
                break
                
    if not s1_code:
        print("Lỗi: Không tìm thấy cell chứa lệnh chạy stage_01_diarize.py trong 01_v01_diarization.ipynb!")
        return
        
    # Tìm cell tương ứng trong Full Notebook
    replace_idx = -1
    for i, cell in enumerate(full_nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "stage_01_diarize.py" in src and "run_logged" in src:
                replace_idx = i
                break
                
    if replace_idx == -1:
        print("Lỗi: Không tìm thấy cell chứa lệnh chạy stage_01_diarize.py trong Full Notebook!")
        return
        
    # Cập nhật cell chạy
    full_nb["cells"][replace_idx] = s1_code
    
    # Cập nhật thêm title markdown của notebook để user dễ nhận biết
    full_nb["cells"][0]["source"][0] = "# Sommelier Kaggle Full Trace Run - (Updated Stage 1 from Incremental)\n"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_nb, f, indent=1)
        
    print(f"✅ Đã tạo thành công phiên bản đầy đủ cập nhật Stage 1 tại: {out_path}")

if __name__ == "__main__":
    main()
