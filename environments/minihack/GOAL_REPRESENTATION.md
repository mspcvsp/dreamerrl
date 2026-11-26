# Room 5x5 offline goal representations

Compare the frozen v1.1 goal autoencoder with two fresh copies of its
architecture. The matched absolute control encodes the actual latent state
at t+K. The displacement model encodes the difference between that same
future state and the state at t. Its decoded difference is added to the
initial state when calculating reconstruction error. K is the frozen
policy's existing four-action goal interval. Both fresh models keep the
existing reconstruction and code-usage objective. They use identical
per-seed initialization, optimizer budgets and validation selection.

The pinned dataset contains one contiguous sequence of visited states per
Room 5x5 audit level. A window must contain all K+1 frames inside a single
recorded episode. Fit, validation and test are disjoint by level. Training
receives RSSM tensors only. Position first enters when offline ridge probes
are fit, after the representations have been frozen.

The report measures comparable feature MSE, hard-code use, centered decoded
rank, prototype cosine, and level-disjoint spatial probes. The in-horizon
distance compares every code's candidate with recorded successors of the
same start. It measures proximity under the archived behavior distribution.
The experiment leaves the RSSM, actor and critic frozen, so worker
controllability is outside its scope.

Run from the clean tagged representation worktree with its locked legacy
uv profile. Pass the main checkout and a fresh output path under main runs. The run records the exact invocation.
Corridor R2 stays deferred.
