# Synthetic point-in-time recovery note

Historical equity-universe reconstruction requires time-aware identifiers.
Symbols can be reused, exchanges can change, and a current company name must
not be projected backwards into old holdings. Every observation therefore
keeps an effective date, source timestamp and stable internal identifier.

A recovery pipeline first joins the dated universe to the contemporaneous
security master. Unresolved records are quarantined rather than silently
matched by a modern ticker. Corporate actions are then applied in event-date
order. Delisting, acquisition and successor fields are evidence attributes;
none may be inferred merely because a price series ends.

The test fixture is deliberately repetitive enough to surround a small fact
inserted by the benchmark controller. It contains no real customer holdings,
credentials, proprietary labels or private evaluation answers. Its purpose is
only to test whether a model can retrieve the explicit inserted date from
irrelevant process text.

For auditability, each accepted mapping stores the raw source reference, the
normalization rule and a confidence status. Reprocessing must be deterministic:
identical fixture revision and rules produce identical mappings. Any manual
override is append-only and records the operator, reason and prior value.

Coverage is reported by source and date cohort. A high aggregate match rate
does not justify accepting a weak subgroup. Downstream research consumes only
records that passed the selected confidence threshold; quarantined rows stay
visible in operational reporting.
