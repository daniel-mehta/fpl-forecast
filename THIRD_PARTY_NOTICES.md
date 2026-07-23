# Third-Party Notices

This project depends on third-party open source software. Daniel Mehta's original source code is
licensed under `AGPL-3.0-only`, but third-party components remain under their respective licences.
This file is a practical dependency notice, not a complete legal inventory of every transitive file.

## Python Runtime And Development Dependencies

Important direct dependencies from `pyproject.toml` and `uv.lock`:

| Package | Locked Version | Licence Summary | Source / Licence |
| --- | ---: | --- | --- |
| DuckDB Python | 1.5.4 | MIT | https://github.com/duckdb/duckdb-python |
| HTTPX | 0.28.1 | BSD-3-Clause | https://github.com/encode/httpx |
| pandas | 3.0.3 | BSD-3-Clause | https://github.com/pandas-dev/pandas |
| PyArrow / Apache Arrow | 25.0.0 | Apache-2.0 | https://arrow.apache.org/ |
| Pydantic | 2.13.4 | MIT | https://github.com/pydantic/pydantic |
| Rich | 15.0.0 | MIT | https://github.com/Textualize/rich |
| SciPy | 1.18.0 | BSD-3-Clause, with bundled numerical-library notices | https://github.com/scipy/scipy |
| Typer | 0.27.0 | MIT | https://github.com/fastapi/typer |
| pytest | 9.1.1 | MIT | https://github.com/pytest-dev/pytest |
| Ruff | 0.15.22 | MIT | https://github.com/astral-sh/ruff |

Notable transitive or bundled Python components observed in the locked environment include NumPy
under BSD-3-Clause plus permissive bundled notices, certifi under MPL-2.0, python-dateutil under
BSD/Apache terms, and SciPy wheel notices for OpenBLAS/LAPACK under BSD-family terms plus GCC
runtime libraries such as `libgfortran` under `GPL-3.0-or-later WITH GCC-exception-3.1` and
`libquadmath` under `LGPL-2.1-or-later`.

## Frontend Dependencies

Important direct dependencies from `frontend/package.json` and `frontend/package-lock.json`:

| Package | Locked Version | Licence Summary | Source / Licence |
| --- | ---: | --- | --- |
| Papa Parse | 5.5.4 | MIT | https://github.com/mholt/PapaParse |
| React | 19.2.8 | MIT | https://github.com/facebook/react |
| React DOM | 19.2.8 | MIT | https://github.com/facebook/react |
| Vite | 7.3.6 | MIT | https://github.com/vitejs/vite |
| TypeScript | 5.8.3 | Apache-2.0 | https://github.com/microsoft/TypeScript |
| ESLint | 9.39.5 | MIT | https://github.com/eslint/eslint |
| `@vitejs/plugin-react` | 4.7.0 | MIT | https://github.com/vitejs/vite-plugin-react |
| `typescript-eslint` | 8.65.0 | MIT | https://github.com/typescript-eslint/typescript-eslint |
| React type packages | 19.x | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped |

The frontend lockfile contains mostly MIT packages, with Apache-2.0, BSD-2-Clause,
BSD-3-Clause, ISC, BlueOak-1.0.0, Python-2.0, and CC-BY-4.0 also appearing in transitive packages.
The practical scan did not find GPL-2.0-only, SSPL, Commons Clause, Business Source License,
PolyForm, noncommercial, or custom/unknown critical direct dependency blockers.

## Data And External Services

See [DATA_NOTICE.md](DATA_NOTICE.md) for the separate treatment of official FPL content, Vaastav
historical data, Premier League material, trademarks, and database rights.

## Notice Preservation

When redistributing bundled dependencies or binary artifacts, preserve the copyright, licence, and
notice files shipped by those dependencies. The links above point to authoritative upstream package
sources; lockfiles record the exact versions used by this repository.
