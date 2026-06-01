import json
import re
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Sử dụng: python tools/inject_beautiful_html.py <đường_dẫn_thư_mục_run_full>")
        sys.exit(1)
        
    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.exists():
        print(f"Lỗi: Thư mục {run_dir} không tồn tại!")
        sys.exit(1)
        
    # Read the beautiful template (index.html at root)
    root_dir = Path(__file__).parent.parent
    template_path = root_dir / "index.html"
    if not template_path.exists():
        print(f"Lỗi: Không tìm thấy file {template_path} để làm template.")
        sys.exit(1)
        
    html = template_path.read_text(encoding="utf-8")
    
    # Load all available stages
    stage_data = {}
    stages = [
        ("01_diarization", run_dir / "01_diarization" / "diarization.json"),
        ("04_asr", run_dir / "04_asr" / "transcript.json"),
        ("05_export", run_dir / "05_export" / "final" / "data_audio.json")
    ]
    
    for stage_name, json_path in stages:
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            segments = data.get("segments", data) if isinstance(data, dict) else data
            
            # Fix audio paths for export stage
            if stage_name == "05_export":
                for seg in segments:
                    if "audio_file" in seg and seg["audio_file"].startswith("data_audio/"):
                        seg["audio_file"] = f"05_export/final/{seg['audio_file']}"
            
            # Ensure every segment has an index
            for i, seg in enumerate(segments):
                if "index" not in seg:
                    seg["index"] = f"{i:05d}"
                    
            stage_data[stage_name] = segments
            
    if not stage_data:
        print(f"Lỗi: Không tìm thấy file JSON nào trong các stage của {run_dir}")
        sys.exit(1)
        
    stage_data_json = json.dumps(stage_data, ensure_ascii=False)
    
    # Replace the SEGMENTS block with STAGE_DATA
    replacement = f'const STAGE_DATA = {stage_data_json};\nlet current_stage = "05_export" in STAGE_DATA ? "05_export" : Object.keys(STAGE_DATA)[0];\nlet SEGMENTS = STAGE_DATA[current_stage] || [];\n'
    
    # Try the old SEGMENTS way first
    html, n = re.subn(r'^const SEGMENTS = \[.*\].*$', replacement, html, count=1, flags=re.MULTILINE)
    
    if n == 0:
        # Try replacing the fallback block instead
        fallback_pattern = r'// Initialize Stage Selector.*?\n\}'
        html, n2 = re.subn(fallback_pattern, replacement, html, count=1, flags=re.DOTALL)
        if n2 == 0:
            print("⚠️ Cảnh báo: Không tìm thấy chỗ để inject STAGE_DATA trong template.")
        
    # Replace the title (Run Full 8 -> Run Full X)
    run_name = run_dir.name
    html = re.sub(r'Run Full \d+', run_name.title(), html, flags=re.IGNORECASE)
    
    # Save the new file INSIDE the run_full directory so relative audio paths work
    out_path = run_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    
    print(f"✅ Đã clone giao diện index.html thành công cho: {run_name}")
    print(f"👉 File được lưu tại: {out_path}")
    print(f"Bạn có thể nhấp đúp vào file này để xem báo cáo siêu đẹp của {run_name}!")

if __name__ == "__main__":
    main()
