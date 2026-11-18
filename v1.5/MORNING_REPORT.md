# Morning report — v1.5 (2026-11-19)

## Result

Long-route author FP is **4.1% (2/49)**. Overall author FP 7.5% is dragged by 4 short poems,
which are routed to v1.2. MiMo-new is **97.1%**; external-68 is a short set and is not a v1.5
metric.

## What I added

- 1,990 extra modern human poems (real titles, 160+46) -> 4,924 human poems total.
- MiMo on/off, DeepSeek long, AI诗, 同族, 拟目 in the AI pool.
- Real-title consistency across train/eval; the old title-leak numbers are retracted.

## Not final

Do not tag v1.5 or publish it as 5% overall. The long route is 4.1%; the short route still
uses v1.2 (author short: 2/4 flagged; three-state reduces hard errors).

## Next candidate work

- Add another 2k modern humans and check if long-route FP reaches 0 without MiMo loss.
- Improve the short route separately; v1.5 does not solve short poems.
