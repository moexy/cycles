# cyclonaut — working conventions

## Write in Markdown

Default to Markdown for every written deliverable — artifacts, reports, handoffs, plans,
notes. Do not reach for HTML because a page "deserves" design treatment. Only use HTML when
the task genuinely cannot be expressed in Markdown (an interactive tool, a chart, a live
dashboard) or when explicitly asked for it.

## Artifacts are local first

Write the artifact source file to a durable path inside the repo and commit it **before**
calling Artifact. Publish from that path.

Never publish from the session scratchpad (`/private/tmp/claude-501/.../scratchpad/`). It is
session-scoped and gets cleaned, so the published page becomes effectively un-updatable —
republishing requires the same source file — and the deliverable leaves no version-controlled
record.

The scratchpad is for throwaway probes and intermediate data only.

Both rules apply to any generated deliverable that may need revising later, not only published
artifacts. A Markdown artifact takes its name from its filename, so name the file the way the
page should be named.
