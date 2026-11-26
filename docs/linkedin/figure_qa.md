# LinkedIn figure profile and visual QA

Build a fresh LinkedIn-only output with `bash docs/linkedin/build_figures.sh /absolute/main/runs/new_linkedin_output`. Run `docs/linkedin/check_preview.py` in the locked publication uv environment with the output directory as its argument. It uses Chromium for local preview proof captures at 400, 600 and 800 CSS pixels. Add `?width=400`, `?width=600`, or `?width=800` to the HTML URL to select a width. The optional `&figures=1` view isolates figures and captions in article order. The local page approximates display widths. LinkedIn controls its own cropping, compression and spacing.

| Preset | PNG dimensions | Figures |
|---|---:|---|
| Cover | 2000 × 600 (10:3) | Beyond Reconstruction |
| Standard | 1600 × 900 (16:9) | Conductor, architecture, maze, bottleneck, intervention, training signal |
| Detailed | 1600 × 1200 (4:3) | CAGE2 opponent results and archived forecasting/common support. Extra height separates groups and interval details. |

The [shared profile](../publication_figures/linkedin_style.json) records the typography tiers, presets, export DPI, preview widths, contrast thresholds and approved color pairs. The [palette](../publication_figures/palette.json) retains its established colors and adds the two approved tints. Contrast validation uses WCAG relative luminance as a design check, with 4.5:1 for meaningful text, 7:1 preferred for essential text and 3:1 for meaningful indicators. The validator accounts for foreground alpha over its known surface. The LinkedIn designs use solid surfaces. Navy on mist narrowly misses 7:1 and remains above the text minimum.

The conductor composes all four steps horizontally with a return arrow in a dedicated lane. Labels stay on pale surfaces or use ivory on navy. Bars and evidence indicators use dark accent variants so they stand out from ivory. Exact ratios used by each figure are recorded in the generated figure provenance. Label bounds and label-to-label overlap are checked at native size. The archive-only improvement and the separate common-support limitation remain visually distinct.

The earlier article and scientific evidence are preserved. The previous Scout deck and its generated figures are outside this revision.

## Review results

Chromium proof captures used 400, 600 and 800 CSS px content columns. The mobile review prompted wider workflow cards, less crowded marker stacks and corrected proof viewport alignment. The release retains 4:3 for the two denser CAGE2 figures. Their sample and interval details remain secondary to the main results.

| Foreground on background | Measured ratio |
|---|---:|
| Navy on ivory, or ivory on navy | 11.600:1 |
| Navy on misty blue | 6.963:1 |
| Navy on pale teal | 10.849:1 |
| Navy on pale orange | 10.677:1 |
| Dark teal indicator on ivory | 5.766:1 |
| Dark orange indicator on ivory | 6.034:1 |

The browser proofs and per-image geometry are under the release output’s `qa/`. Figure titles, value annotations and margins were reviewed at each target width. Rounded chart labels favor quick reading, with full-precision values and intervals retained in the existing evidence snapshot. The very wide cover uses a shorter subtitle and flow labels at mobile scale. Display cropping and compression on LinkedIn remain platform-dependent.
