# Publication bundle build and handoff

The article, slides, guide, visual identity and figure sources are tracked. The build writes a new ignored output directory under the main checkout’s `runs/`. It reads the small [aggregate snapshot](../publication_figures/evidence.json); local checkpoints, replay files and the simulator are unnecessary for building these materials.

| Source | Result |
|---|---|
| [LinkedIn draft](../linkedin/article.md) and `prepare_article.py` | `linkedin/article.md` with the cover and seven figure placements, captions and alt text |
| [Shared palette](../publication_figures/palette.json), [figure register](../publication_figures/figure_manifest.json), and [renderer](../publication_figures/generate_figures.py) | Eight LinkedIn and ten Scout graphics, each as editable SVG, vector PDF and 320 DPI PNG |
| [Scout Markdown](slides.md), [Beamer theme](theme.tex), and [PowerPoint reference](reference.pptx) | `presentation/scout_presentation.pdf` and `presentation/scout_presentation.pptx` |
| [Presenter guide](presenter_guide.md) and [saved-results demo](../publication_figures/read_saved_results.py) | A ready-to-teach lesson plan and CPU-only, dependency-free terminal walkthrough |
| `build.sh`, `manifest.py`, and `render_qa.py` | `build_manifest.json`, `figure_provenance.json` and the visual QA proof directory |

## Rebuild

Use the existing locked Matplotlib environment and system Pandoc, XeLaTeX/Beamer, Poppler, librsvg and Fontconfig. Inter is preferred. The builder selects DejaVu Sans if Inter is absent and records the selected font. LibreOffice is used for PowerPoint QA where available. No package installation occurs during the build.

```bash
bash docs/presentation/build.sh "$PWD/runs/publication_example"
uv run --locked --no-sync python docs/presentation/render_qa.py "$PWD/runs/publication_example"
```

From a linked worktree use the actual main checkout's `runs/` location and point `uv run --project` at the main locked environment for the QA command. The builder detects the main checkout automatically. A different locked project or pre-provisioned environment can be selected via `PUBLICATION_UV_PROJECT` or `PUBLICATION_ENVIRONMENT`. Keep build output directories distinct from frozen experiment directories.

Pandoc reads one Markdown deck. Beamer uses vector figure PDFs converted from text-preserving SVG. The editable PowerPoint uses 320 DPI PNG illustrations, editable headings and explanatory text, an explicit widescreen layout, and original alternative descriptions for embedded graphics. The PPTX transformer preserves the source content while keeping each explanatory line with its image. The [build manifest schema](manifest.py) records versions, source hashes and hashes of generated files.

To regenerate the reference deck from the Pandoc default, first export it with `pandoc --print-default-data-file=reference.pptx` into a temporary file, then pass that source and the destination to `style_pptx.py` with the selected font and `--reference`.

## Evidence and release

The snapshot preserves full precision for the plotted quantities. The [extraction script](../publication_figures/extract_evidence.py) can reproduce it when pointed at the main checkout with the original ignored run artifacts. Compare the generated snapshot byte-for-byte with the tracked copy. No replay or additional fit is required.

The [release review](release_readiness.md) lists portability, privacy and verification boundaries. The [visual QA record](visual_qa.md) records the reviewed output and corrections. The [closeout claims matrix](../closeout/claims.md) documents which experimental statements are established, local or unresolved.
