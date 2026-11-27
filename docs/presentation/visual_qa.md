# Publication visual QA

The proof sheet review covers all 15 Beamer slides and all 15 PowerPoint slides rendered through LibreOffice. LinkedIn cover and editorial graphics are checked at mobile 400 px and desktop 800 px widths. The automated check confirms the page counts, expected rendered headings, text bounding boxes within the page, and the presence of editable text on each PowerPoint slide. The SVG sources retain text and the figure PDF is a vector export.

Proof corrections included:

- Kept the teaching line with its image on all PowerPoint slides. The two exports now have the same fifteen-slide sequence.
- Shortened the mobile maze headline, increased right margins on value annotations, and broke up long workflow and career labels before export.
- Used a dark title, clue activity and conclusion consistently. Improved hyperlink contrast on the dark close.
- Converted the vector PDF from SVG with librsvg to avoid a font-type warning on the Inter OTF export.
- Kept chart comparisons readable with direct labels, symbols and line styles, and identified conceptual drawings in captions.

The generated `qa/` directory contains the actual slide renders, desktop and mobile previews and `automated_checks.json`. The PowerPoint proof demonstrates rendering in LibreOffice. Microsoft PowerPoint was unavailable on the build host.

## Human–AI narrative revision

The revised proofs were generated for all 15 slides in both formats. The Scout conductor’s feedback line was moved clear of its caption and the tool names were added to the diagram. The professional conductor figure was reviewed at 400 and 800 px. The other article previews were checked against the earlier visually approved exports. The [consistency matrix](narrative_consistency.md) records the attribution boundary and reviews the captions and descriptions. Revised outputs use a fresh ignored directory so the earlier release and its manifests remain available.

## Scout lesson revision

Build directory `runs/dreamerv3_scout_learning_final_20261126/`. Used the existing Pandoc workflow with `--scout-only`. Reviewed all 15 Beamer pages and all 15 LibreOffice-rendered PowerPoint pages at 1280 × 720, including figure labels, captions, margins, and projected text contrast. The final revisions clarified the turn-choice label, emphasized the sampled-summary result, and darkened the conductor diagram’s return arrow. Unchanged pages matched the reviewed renders byte for byte. Automated checks verified page count, heading editability, and PDF text bounds. The PPTX contains alternative descriptions for all 11 embedded figures.

Reviewed for two audiences. A Scout new to machine learning can predict a turn before seeing the plot and participate from a seat. A Scout interested in engineering can use the guide to discuss fair probe tests, one-draw accuracy bounds, and why a readable clue does not establish that the agent uses it. The saved-results walkthrough ran using Python 3 and the tracked evidence snapshot. Projection in a troop meeting room and rendering in Microsoft PowerPoint were outside this local review.
