# Two sample contracts

The project brief asks for two sample contracts, with the risky sentences extracted and flagged. These are
those two, with the tool's output for each.

| # | Contract | Type | Your side | Source |
|---|---|---|---|---|
| 1 | `contracts/01_hop_dong_lao_dong_xac_dinh_thoi_han.docx`: fixed-term labour contract | `labour` | Người lao động | Free public template linked from cogover.com ("Mẫu hợp đồng lao động xác định thời hạn"), downloaded 2026-11-13 |
| 2 | `contracts/02_hop_dong_hop_tac_kinh_doanh.docx`: business cooperation agreement | `commercial` | Bên B | Free public template "Mẫu hợp đồng thỏa thuận hợp tác kinh doanh", downloaded 2026-11-11 |

Neither contract appears in the training data (CUAD/LEDGAR are English).

## Reproduce

```bash
python -m lca.cli samples/contracts/01_hop_dong_lao_dong_xac_dinh_thoi_han.docx \
    --type labour --party "Người lao động" --out samples/output
python -m lca.cli samples/contracts/02_hop_dong_hop_tac_kinh_doanh.docx \
    --type commercial --party "Bên B" --out samples/output
```

Each run writes `*_danh_dau.docx` (the original file with risky sentences highlighted red and a Word comment
explaining each one) and `*_bao_cao.csv` (one row per finding: location, sentence, risk group, entities,
level, explanation).

## What the tool flags

**Contract 1 (labour), 19 findings: 1 Cao, 6 Trung bình, 12 Thấp**

| Level | Where | Sentence | Why |
|---|---|---|---|
| Cao | Điều 3 | "…nhân viên phải hoàn trả lại 100% phí đào tạo…" | Full repayment of training costs if the employee leaves |
| Trung bình | Điều 5 | "…người sử dụng lao động có quyền đơn phương chấm dứt hợp đồng…" | Employer's unilateral termination |
| Trung bình | Điều 4 | "Có quyền đòi bồi thường… nếu người lao động vi phạm…" | Employer can claim compensation |
| Thấp | Điều 3 | "Bồi thường vi phạm và vật chất theo quy chế, nội quy của Công ty…" | Liability set by internal company rules |

**Contract 2 (commercial), 4 findings: 1 Cao, 2 Trung bình, 1 Thấp**

| Level | Where | Sentence | Why |
|---|---|---|---|
| Cao | Điều 8.2 | "…phải bồi thường toàn bộ thiệt hại xảy ra và chịu phạt vi phạm bằng 10% giá trị của thỏa thuận." | Uncapped damages, and a 10% penalty above the 8% cap (Điều 301 Luật Thương mại 2005) |
| Trung bình | Điều 8.3 | "Đảm bảo bí mật mọi thông tin liên quan tới quá trình sản xuất kinh doanh." | Confidentiality |
| Trung bình | Điều 9.1 | "Thỏa thuận chấm dứt khi hết thời hạn…" | Termination |
| Thấp | Điều 9.1 | "Khi kết thúc Thỏa thuận, hai bên sẽ làm biên bản thanh lý thỏa thuận." | Termination (similarity only) |

## Known false alarms

Some Thấp findings in contract 1 come from the similarity method alone and are wrong. For example, "Hình thức
trả lương: Chuyển khoản…" is labelled as a penalty. Thấp is the "worth a glance" level: the tool is
recall-first, so it prefers an extra Thấp to a missed risk.

*Công cụ chỉ mang tính hỗ trợ sàng lọc, không thay thế tư vấn pháp lý.*
