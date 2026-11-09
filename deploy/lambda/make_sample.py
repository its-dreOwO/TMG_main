"""Make sample Vietnamese contract PDFs (1 and 30 pages) with a font that has Vietnamese glyphs."""
import sys
from pathlib import Path

import pymupdf

FONT = "/usr/share/fonts/noto/NotoSans-Regular.ttf"
LINES = ["Điều 8. Phạt vi phạm",
         "Bên B phải chịu phạt 12% giá trị hợp đồng nếu giao hàng chậm.",
         "Bên B bồi thường toàn bộ thiệt hại phát sinh, không giới hạn.",
         "Bên B có nghĩa vụ bảo mật thông tin trong 02 năm sau khi chấm dứt hợp đồng.",
         "Hợp đồng có hiệu lực kể từ ngày ký."]


def make(pages: int, out: Path) -> None:
    doc = pymupdf.open()
    for p in range(pages):
        page = doc.new_page()
        page.insert_font(fontname="noto", fontfile=FONT)
        # Number every line: identical pages would all be stripped as repeated headers/footers.
        lines = [f"{p * 30 + i + 1}. {line}" for i, line in enumerate(LINES * 6)]
        page.insert_textbox(pymupdf.Rect(60, 60, 540, 780), "\n".join(lines), fontsize=10, fontname="noto")
    doc.save(out)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    make(1, out / "smoke_1.pdf")
    make(30, out / "smoke_30.pdf")
