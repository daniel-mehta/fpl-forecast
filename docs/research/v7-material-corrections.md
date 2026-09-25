# Material v7 manuscript corrections for the author

The attached `Forecasts to Decisions v7.pdf` is the manuscript authority. No editable manuscript
source is present in this repository, and the PDF has not been changed. Apply only these replacements
to the editable source when available, then verify the rendered pages.

| PDF page / location | Current wording | Exact replacement | Reason |
| --- | --- | --- | --- |
| 7, Data sources and scope, historical-panel paragraph | `83,835 player-fixture rows` | `83,835 entity-fixture rows` | The count includes 322 Assistant Manager entity records, excluded from player models. |
| 7, same paragraph, next sentence | `Player-fixture grain was retained because a player can appear more than once in a double Gameweek.` | `Fixture-level grain was retained because a football player can appear more than once in a double Gameweek.` | Keeps the explanation of doubles without calling the entire panel football-player rows. The later 322-record sentence remains accurate. |
| 22, Limitations and prospective protocol, opening sentence of final paragraph | `The 2026-27 season will provide a prospective evaluation.` | `The 2026-27 prospective evaluation is underway.` | Official pre-deadline forecasts have been frozen and deployed for GW1–GW6. Keep the subsequent protocol and missed-deadline/no-backfill rule. |
| 24, Conclusion, opening sentence | `The next stage is prospective evaluation during the 2026-27 season.` | `Prospective forecast publication is underway during the 2026-27 season.` | Corrects tense while leaving outcome evaluation prospective. The following assessment sentence remains future-oriented. |
| 24, Code and Data Availability, whole paragraph | `The source code, configuration files, validation checks, evidence manifests, derived publication assets, and instructions supporting this study are available at https://github.com/daniel-mehta/fpl-forecast. Raw and normalised third-party data are not redistributed in the repository; their sources and the processing steps used to construct the research panel are documented, subject to continued source availability and applicable data terms.` | `The source code, configuration files, validation checks, aggregate evidence manifests and publication assets, source-input inventory, research schema, and rights-bounded reproduction instructions are available at https://github.com/daniel-mehta/fpl-forecast. Raw and normalised third-party data and row-level research artifacts are excluded from the main research bundle. Access to the original sources remains subject to availability and applicable data rights; the separate public forecast branch contains sanitised operational forecasts.` | Matches the Oct. 1 rights-safe public bundle without implying that the repository alone supplies all experimental input rows. Recheck this paragraph against the final public Git state before submission. |

No other prospective downgrade is supported by the GitHub Actions record. In particular, retain
the p. 23 rule that missed deadlines and operational failures are recorded rather than backfilled.
The six registered GW1-GW6 publication runs establish timely publication, not scored predictive accuracy.
