# Offline variance-aware goals in Room 5x5

This study inherits the pinned v1.1 checkpoint, four-action archive pairs,
level-disjoint splits, and five initialization seeds from the offline
representation study. Only fresh goal autoencoders learn. The original
checkpoint provides a frozen historical reference. Each arm uses the
same categorical code count, hidden width, loss terms, optimizer budget,
and validation-selection rule. Full-width arms share weights at
initialization within each seed. The h-only arm has fewer parameters
because its input and output use h alone.

Fit means and population standard deviations use fitting levels only.
Every per-feature scale is floored at 0.01 in latent units, including
constant and nearly constant coordinates. A trains in native units.
B trains on normalized absolute states, C on normalized displacements,
and D on normalized absolute h. Original-unit scoring inverts the
transformation and anchors C to the observed start. Compare D with
the other arms using the common h-only metric. Each arm has its own
fit-mean baseline. Normalized scores are relative to each target's scale.

Position enters only in ridge probes after fitting. Probe baselines
use start position and timestep. The new goal representation must lower
mean X/Y test displacement RMSE versus both absolute references, retain
at least two effective codes, and beat its own normalized mean
reconstruction baseline on every paired initialization. Code geometry
and proximity to recorded future states are descriptive offline measures.
Corridor R2 remains deferred.

From the clean annotated checkout with the locked legacy profile,
invoke `environments/minihack/normalized_goals.py` with the main checkout
path and a new output directory under main runs. The detached run records
its exact command, input fingerprints, and split membership.
