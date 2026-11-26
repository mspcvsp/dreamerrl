"""WCAG relative luminance for opaque surfaces and composited foregrounds."""
from matplotlib.colors import to_rgba


def luminance(rgb):
    linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in rgb]
    return sum(c * w for c, w in zip(linear, (.2126, .7152, .0722), strict=True))


def contrast(foreground, background):
    *back, alpha = to_rgba(background)
    if alpha != 1:
        raise ValueError("Composite the background over its actual surface first")
    *front, alpha = to_rgba(foreground)
    front = [alpha*f + (1-alpha)*b for f, b in zip(front, back, strict=True)]
    hi, lo = sorted((luminance(front), luminance(back)), reverse=True)
    return (hi+.05)/(lo+.05)


def require_contrast(foreground, background, minimum):
    ratio = contrast(foreground, background)
    if ratio < minimum:
        raise ValueError(f"{foreground} on {background}: {ratio:.2f}:1 below {minimum}:1")
    return ratio
