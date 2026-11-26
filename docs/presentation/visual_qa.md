# Publication visual QA

The proof sheet review covers all 15 Beamer slides and all 15 PowerPoint slides rendered through LibreOffice. LinkedIn cover and editorial graphics are checked at mobile 400 px and desktop 800 px widths. The automated check confirms the page counts, expected rendered headings, text bounding boxes within the page, and the presence of editable text on each PowerPoint slide. The SVG sources retain text and the figure PDF is a vector export.

Proof corrections included:

- Kept the teaching line with its image on all PowerPoint slides. The two exports now have the same fifteen-slide sequence.
- Shortened the mobile maze headline, increased right margins on value annotations, and broke up long workflow and career labels before export.
- Used a dark title, clue activity and conclusion consistently. Improved hyperlink contrast on the dark close.
- Converted the vector PDF from SVG with librsvg to avoid a font-type warning on the Inter OTF export.
- Kept chart comparisons readable with direct labels, symbols and line styles, and identified conceptual drawings in captions.

The generated `qa/` directory contains the actual slide renders, desktop and mobile previews and `automated_checks.json`. The PowerPoint proof demonstrates rendering in LibreOffice. Microsoft PowerPoint was unavailable on the build host.
