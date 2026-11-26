# Repository publication readiness

**Disposition:** Ready for owner review on the local publication branch. A remote merge or push requires explicit approval.

## New reader path

- The README separates a dependency-free CPU demonstration, publication rendering, frozen diagnostics and full training. The demonstration reads aggregate data only.
- The original checkpoint provenance, immutable baseline tag and separate MiniHack/CAGE2 uv profiles remain documented. External readers can build the publication bundle from the tracked aggregate snapshot without access to frozen weights.
- The publication branch descends from the completed diagnostics branch, with the environment/model adapters, tests, schemas and closeout documents in its history. Main remains separate. Review the branch diff and maintain a separate release decision.

## Licensing and distribution

The project and pinned upstream CAGE2 repository carry MIT licenses. Keep upstream license and attribution notices when distributing their material. The figures are authored from aggregate project measurements or reproducible conceptual drawings. The repository links to the original DreamerV3 paper and source, CAGE2, and MiniHack. The editable reference deck is a restyled Pandoc default template.

The primary deliverables are build outputs under ignored `runs/`. Source, aggregate values, the palette and the small PowerPoint reference template are Git-tracked. Continue to distribute repository data and assets separately from the locally preserved weights and raw simulation artifacts.

## Pre-publication review items

- A current-tree scan checked common credential-shaped strings and sensitive filenames. It found no tracked checkpoint weights or obvious secret matches. This is a scoped check rather than a full Git-history security audit.
- Historical training/reproduction records contain original absolute paths, host identifiers, and user-confirmed backup metadata. The public-facing materials refer to portable repository-relative paths. Review historical content and intended audience before publishing. Preserve the immutable evidence and tags while making any separate distribution decision.
- Full reproduction depends on checkpoints and observations that live in ignored local storage. Readers can run the published demonstration, visualize the aggregates, or read the runtime contract without those inputs. Test a clean clone on the intended release host before claiming general platform compatibility.
- The verified publication build used Linux ARM64. Its system publication tools do not change `uv.lock` or the learner. A reader who lacks LibreOffice can still build an editable deck with Pandoc, and can render it separately on a supported host.
- Scientific claims are scoped by the evidence matrix. The archived CAGE2 comparison is local to the specified protocol. Common-support failure left matched-history recoverability unresolved.
