# Publication bundle build and handoff

The article, slides, guide, visual identity and figure sources are tracked. The build writes a new ignored output directory under the main checkout’s `runs/`. It reads the small [aggregate snapshot](../publication_figures/evidence.json); local checkpoints, replay files and the simulator are unnecessary for building these materials.

| Source | Result |
|---|---|
| [LinkedIn draft](../linkedin/article.md) and `prepare_article.py` | `linkedin/article.md` with the cover and eight figure placements, captions and alt text |
| [Shared palette](../publication_figures/palette.json), [figure register](../publication_figures/figure_manifest.json), and [renderer](../publication_figures/generate_figures.py) | Nine LinkedIn and eleven Scout graphics, each as editable SVG, vector PDF and 320 DPI PNG |
| [Scout Markdown](slides.md), [Beamer theme](theme.tex), and [PowerPoint reference](reference.pptx) | `presentation/scout_presentation.pdf` and `presentation/scout_presentation.pptx` |
| [Presenter guide](presenter_guide.md) and [saved-results demo](../publication_figures/read_saved_results.py) | A ready-to-teach lesson plan and CPU-only, dependency-free terminal walkthrough |
| `build.sh`, `manifest.py`, and `render_qa.py` | `build_manifest.json`, `figure_provenance.json` and the visual QA proof directory |

## First-build setup

Use the repository’s existing Python version and lockfiles. From the main checkout, provision a separate publication environment so the training environment stays untouched. If the platform requires the pinned submodule or native wheel preparation for locked sync, follow the applicable prerequisites in the [uv environment guide](../uv_environment.md) first.

```bash
export PUBLICATION_UV_PROJECT="$PWD"
export PUBLICATION_ENVIRONMENT="$PWD/runs/.publication-venv"
UV_PROJECT_ENVIRONMENT="$PUBLICATION_ENVIRONMENT" uv sync --locked --project "$PUBLICATION_UV_PROJECT"
```

The lock provides Matplotlib for figure rendering and Pillow for image and QA rendering. The verified environment resolved Matplotlib 3.10.9 and Pillow 12.3.0. The builder uses `uv run --locked --no-sync` and expects a provisioned environment.

| System requirement | Role and verified reference |
|---|---|
| Pandoc | Markdown conversion to Beamer and PPTX. Verified with 3.14 on Linux ARM64. |
| XeLaTeX and Beamer | PDF slide export. Verified with XeTeX 0.999995, TeX Live 2023 and the Beamer class from that distribution. |
| Poppler | PDF rendering, text extraction and inspection via `pdftoppm`, `pdftotext`, `pdfinfo` and `pdffonts`. |
| librsvg | Vector PDF conversion via `rsvg-convert`. Verified with 2.58.0. |
| Fontconfig and Inter | `fc-match` resolves the installed Inter faces. DejaVu Sans is the defined fallback. Install fonts through your approved system setup. |
| LibreOffice Impress | Optional for export, required for the documented local PPTX rendering check. Verified with LibreOffice 24.2.7.2. |

## Check and rebuild

From the source checkout, with the exports above available if needed:

```bash
bash docs/presentation/build.sh --check
bash docs/presentation/build.sh "$PUBLICATION_UV_PROJECT/runs/publication_example"
UV_PROJECT_ENVIRONMENT="$PUBLICATION_ENVIRONMENT" uv run --project "$PUBLICATION_UV_PROJECT" --locked --no-sync \
  python docs/presentation/render_qa.py "$PUBLICATION_UV_PROJECT/runs/publication_example"
```

Add `--scout-only` after the output path to build and check just the Scout deliverables and their figures.

The preflight checks executables, Beamer, Python imports and font resolution. It prints font families, styles, filenames and hashes without generating slides or figures. The [current host verification](font_verification.json) records the Inter regular and bold files resolved by Fontconfig and Matplotlib at verification time. Future builds write their own `font_resolution.json` and embed the record in their build manifest. Font binaries stay local. The prior release manifest remains the original build record.

In a linked worktree, use the main checkout’s absolute path for `PUBLICATION_UV_PROJECT` and for the output under `runs/`. Keep publication and experiment directories distinct.

Pandoc reads one Markdown deck. Beamer uses vector figure PDFs converted from text-preserving SVG. The editable PowerPoint uses 320 DPI PNG illustrations, editable headings and explanatory text, an explicit widescreen layout, and original alternative descriptions for embedded graphics. The PPTX transformer preserves the source content while keeping each explanatory line with its image. The [build manifest schema](manifest.py) records versions, source hashes and hashes of generated files.

To regenerate the reference deck from the Pandoc default, first export it with `pandoc --print-default-data-file=reference.pptx` into a temporary file, then pass that source and the destination to `style_pptx.py` with the selected font and `--reference`.

## Evidence and release

The tracked `docs/publication_figures/evidence.json` snapshot preserves full precision for the plotted quantities. Figure regeneration reads that snapshot and the tracked palette and provenance register. The figure generator requires no original experiment directories, simulator, or weights.

For independent re-extraction, retain the original experimental JSON under the main checkout’s ignored `runs/` tree. The snapshot’s `source_sha256` map gives the canonical relative paths and SHA-256 digests. Preserve the original paths or a documented path mapping, and verify the hashes before using a restored copy. See the [experimental artifact inventory](../closeout/artifacts.md) for provenance.

Keep the owner-approved exports in `runs/dreamerv3_publication_release_20261126/` with their original `build_manifest.json`, `figure_provenance.json` and rendered QA proofs. Preserve a separately managed copy as needed and verify the manifest hashes after transfer. Use a fresh directory for any future build. The [extraction script](../publication_figures/extract_evidence.py) can reproduce it when pointed at the main checkout with the original ignored run artifacts. Compare the generated snapshot byte-for-byte with the tracked copy. No replay or additional fit is required.

The [release review](release_readiness.md) lists portability, privacy and verification boundaries. The [visual QA record](visual_qa.md) records the reviewed output and corrections. The [closeout claims matrix](../closeout/claims.md) documents which experimental statements are established, local or unresolved.
