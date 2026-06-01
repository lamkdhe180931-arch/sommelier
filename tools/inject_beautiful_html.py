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
    
    # Try to load the new segments data from the target run_full
    json_path = run_dir / "05_export" / "final" / "data_audio.json"
    if not json_path.exists():
        json_path = run_dir / "04_asr" / "transcript.json"
    if not json_path.exists():
        json_path = run_dir / "01_diarization" / "diarization.json"
        
    if not json_path.exists():
        print(f"Lỗi: Không tìm thấy file JSON nào trong {run_dir}/05_export/final/, {run_dir}/04_asr/ hoặc {run_dir}/01_diarization/")
        sys.exit(1)
        
    data = json.loads(json_path.read_text(encoding="utf-8"))
    
    # data is either a list of segments or a dict with "segments" key
    segments = data.get("segments", data) if isinstance(data, dict) else data
    
    # Convert to JSON string
    segments_json = json.dumps(segments, ensure_ascii=False)
    
    # Replace the SEGMENTS block (it is on one line or spans multiple lines)
    # We look for "const SEGMENTS = [" up to the matching closing bracket
    html = re.sub(r'const SEGMENTS = \[.*\];?', f'const SEGMENTS = {segments_json};', html, count=1, flags=re.DOTALL)
        
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
