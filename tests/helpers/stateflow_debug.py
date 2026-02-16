def debug_stateflow(pre_hxs, post_hxs, title="Stateflow"):
    """
    Pretty-print PRE/POST stateflow for debugging TBPTT alignment.
    """
    T, B, H = pre_hxs.shape
    print(f"\n=== {title} ===")
    for t in range(T):
        print(f"t={t}")
        print(f"  PRE  h[{t}]: {pre_hxs[t, 0, :5].tolist()} ...")
        print(f"  POST h[{t}]: {post_hxs[t, 0, :5].tolist()} ...")
