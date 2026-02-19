# Important Note: `PolicyEvalInput` Lives in `types.py`, Not in `TrainerState`

This is a reminder for future contributors:

**`PolicyEvalInput` is defined in `types.py` and imported directly from there.
It is *not* a field or nested type inside `TrainerState`.**

Why this matters:

- Several trainer tests (TBPTT, rollout replay, mask propagation) construct `PolicyEvalInput` objects.
- It’s easy to mistakenly try to access it via `trainer.state.PolicyEvalInput`, which will fail.
- The correct import path is:

```python
from lstmppo.types import PolicyEvalInput
```

This keeps the policy API clean and decoupled from the trainer’s internal state.

If you ever refactor the policy or trainer, keep this invariant:

- **Policy input/output dataclasses live in `types.py`.**
- **TrainerState should never own or redefine them.**
- **Trainer should always import them directly from `types.py`.**

This separation ensures:
- clean API boundaries
- easier testing
- no circular dependencies
- consistent construction of evaluation inputs across rollout, TBPTT, and replay paths

If a test ever breaks because of a missing `PolicyEvalInput`, check your imports first.
