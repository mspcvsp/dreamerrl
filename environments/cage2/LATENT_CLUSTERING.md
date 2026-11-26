# Frozen RSSM clustering

Run from the root locked CAGE2 uv environment:

```bash
uv run --locked python environments/cage2/latent_clustering_job.py \
  --run-dir runs/cage2_latent_clustering_host_pilot_20261124
```

The tracked launcher uses detached tmux `cage2-latent-clustering` and a hard
1800-second process-group deadline. Exit status, command, source diff, source and
input hashes, and logs are generated under the output directory. No Dreamer model
is loaded, no environment is stepped, and original data/checkpoints are read-only.
New source is staged for review; its full staged diff and hashes identify the
analysis in addition to HEAD. No new dependency is required.

Reuse the completed frozen-analysis data and exact 216/72/72 episode split.
Verify the checkpoint hash, original tag/commit, locked dependencies, original
analysis source and successful frozen-weight checks. Record current dataset hashes;
the original collection did not independently publish a dataset hash, so these
hashes establish this analysis input, not independent archival attestation.

Standardize h on fitting rows. Fit PCA on standardized fitting h, retain 95%
variance, without whitening. Subtract fitting-only per-timestep h means and
standardize fitting residuals for the clock control. K-means++ uses seeds 11/22/33,
60 maximum Lloyd iterations, K=2..10. Pick initialization by fitting inertia and
K by validation silhouette, earlier K on ties. All test K scores are exploratory.
Silhouette uses fixed random 600-row subsets per split; DB and occupancy use all
split rows. Fit/validation/test initialization stability uses pairwise adjusted
Rand index (ARI). Three fitting-episode bootstrap refits at selected K retain the
original fitting preprocessing; report test-wide and per-test-episode ARI.

Characterize all K on every split with cluster sizes, episode occupancy, timestep
histograms, return quantiles, and total within-cluster feature variance. Temporal
persistence includes adjacent same-cluster rate and streak lengths, never crossing
an episode boundary. Compare persistence against 100 shuffles across episodes
within each timestep, preserving the clock-conditioned occupancy profile.
NMI with timestep and residual clustering help separate clock structure from
additional structure. They do not remove nonlinear time effects or prove semantics.

Outputs: `cluster_quality.csv/json`, `cluster_characterization.json`,
`selected_clusters.json`, per-family model/assignment arrays, fitting preprocessing,
`pca_clusters.png`, `final_report.txt`, and preservation/completion checks.
Plots use common fitting-PCA axes and eight fixed-ID test trajectories.

These are policy-conditional exploratory partitions from one trained agent.
No privileged labels identify compromise or attack phase; return associations are
confounded by time. K-means scores do not establish natural or causal clusters.
Episode bootstrap stability is not independent training-seed uncertainty.

Tests:

```bash
uv run --locked pytest -c /dev/null -p no:cacheprovider -q \
  environments/cage2/test_latent_clustering.py environments/cage2/test_latent_probe.py
```
