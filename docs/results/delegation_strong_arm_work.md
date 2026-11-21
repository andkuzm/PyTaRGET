| Cell | model calls | added context | work (extra input tokens processed) | output tokens |
|---|---|---|---|---|
| P1-1 shop solo r1 | 4 | 15,053 | 36,975 | 3,204 |
| P3-1 shop solo r2 | 4 | 14,919 | 36,631 | 3,244 |
| P1-2 shop S1 | 5 | 4,577 | 16,757 | 776 |
| P2-0b shop S1 r2 | 6 | 6,977 | 25,806 | 1,128 |
| P2-1 shop S2 | 7 | 13,508 | 69,428 | 2,218 |
| P2-2 shop S3 | 7 | 6,550 | 30,192 | 1,443 |
| P2-3 shop S1 strong ctx | 8 | 13,653 | 80,401 | 2,814 |
| P2-4 shop S1 both ctx | 9 | 14,314 | 92,724 | 2,637 |
| P2-5 shop S1 diff check | 8 | 8,041 | 38,543 | 1,967 |
| P1-3 gson_a80 solo | 4 | 9,564 | 22,979 | 804 |
| P1-4 gson_a80 S1 (tool bug) | 34 | 32,690 | 569,095 | 13,867 |
| P2-0 gson_a80 S1 | 40 | 32,505 | 638,397 | 10,399 |
| P1-5 gson_cbad solo | 6 | 21,972 | 84,716 | 1,990 |
| P1-6 gson_cbad S1 | 14 | 19,865 | 124,408 | 3,617 |
| P3-2 gson_cbad S2 | 9 | 14,281 | 92,932 | 1,978 |

fixed start-up context (first call): 43,000 - 43,540 tokens
