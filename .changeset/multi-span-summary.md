---
"prunella": minor
---

Replace top-level firstPartId/lastPartId/collapsed on CompactorSummary with a spans array so a single summary can cover multiple non-consecutive regions. Eliminates the collapsed sentinel hack in favor of honest multi-span representation.
