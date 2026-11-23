# A public release of the Neuronpedia viewer and its data

Written 2026-11-02. The question: what is the best way to publish a public
version of our self-hosted Neuronpedia viewer on GitHub, with a subset of the
data, so that a reader can check the lens results of the paper?

The answer in 1 line: for the review period, host the viewer and the runs on a
free static host behind a password, host the large data as a restricted Zenodo
record with a secret link, and put the code in the same two places as an
anonymized archive. After acceptance, open the same material: a public GitHub
repository, an open Zenodo record (a DOI), and GitHub Pages. The public
neuronpedia.org cannot show our runs. Section 7 holds the review-period plan.
Sections 2 to 6 hold the facts and the post-acceptance plan.

## 1. What we have

| Part | Where | Size | Note |
|------|-------|------|------|
| The viewer | A copy of Neuronpedia at commit `e2573d3`, with 2 local patches (`neuronpedia_up.sh`) | code | Apache 2.0 |
| The runs of the viewer | `data/f10/hop_all/viewer/{model}/runs/*.json` | 307 MB (Gemma-4-31B), 155 MB (Qwen3.5-9B), 304 MB (Qwen3.6-27B); 599 files each | 1 file for each image or group, `JlensExport` version 1 |
| The share rows | `data/f10/hop_all/viewer/{model}/shares.json` | small | 1 row of the table `JlensShare` for each run |
| The thumbnails | `data/f10/hop_all/viewer/img/` | 14 MB, 591 files | Cyclomedia imagery. See section 5 |
| The store parts that the runs come from | `data/f10/hop_all/{model}/`: `text_reads.npy`, `prefix_reads.npy`, `answers/`, `manifest.json` | about 95 MB (Gemma) and 36 MB (Qwen) for the `cool` prompt | `hop_viewer.py` reads these parts only |
| The same parts for the other prompts | `data/f10/hop_all/{model}-{adjective}/`, 15 variants for each of 3 models | about 1.8 GB in total | The F10.19 to F10.24 results |
| The patch top words | `data/f10/hop_all/{model}/patches/ids-*.npy` | about 530 MB for each model | Optional |
| The patch residuals | `data/f10/hop_all/{model}/patches/*.npy` | 47 GB for each model | Do not publish. Give them on request |
| The lenses | `outputs/_monocle/jlens/jpp/{model}/lens.pt` | 6.8 GB (Gemma-4-31B) and 2.1 GB (Qwen3.5-9B) | Neuronpedia's public J++ lenses. Do not host them again. Cite the Hugging Face file and give the sha256 |
| The code | `research/scripts/f10/` (78 files), `research/monocle/jlens_read.py`, `tests/test_hop_*.py` | code | |

A viewer run is small and the store parts behind it are small. The patch
residuals are the only large part, and the viewer does not show them.

## 2. What Neuronpedia gives us

### The share mechanism is a row and a file

A shared run in Neuronpedia is 1 row of `JlensShare` and 1 JSON file at `url`.
The schema says: "A shared jlens (Jacobian Lens) chat/completion run. The heavy
data (messages, meta, and the full unfiltered per-position token stream) is
stored gzipped in S3 at `url`" ([schema.prisma][schema]). The page
`/[modelId]/jlens?shareId=...` reads the row and passes `url` to the browser,
and the browser fetches it: `fetch(share.url, { cache: 'force-cache' })`
([jlens-page-client.tsx][client]). Our local viewer uses exactly this path with
a relative URL. Thus a public copy needs 3 things only: the rows, the files at
a URL the browser can fetch, and the webapp with no inference host.

### The public neuronpedia.org cannot show our runs

The share endpoint `/api/lens/share` accepts a prompt, not a run. The server
runs the inference itself and writes the result to its own S3
([route.ts][share-route]). Neuronpedia's inference takes text only, and our
prompts hold an image. Also the second column of our runs is a contrast, and
the public site labels that column "Logit Lens". Thus a public share on
neuronpedia.org is not possible for our data, and it would mislabel a column if
it were.

### Self-hosting is documented

"All of Neuronpedia's code is open source, under the Apache 2.0 license." The
steps are Postgres 16 with pgvector, `make db-init`, `make webapp-install`,
`make webapp-dev` ([self-hosting docs][selfhost]). The README says "The local
database environment does not have any inference servers connected"
([README][readme]), which is our case. The repo holds a `vercel.json`, thus a
deployment on Vercel is the path the authors use.

### Neuronpedia's own data release is the model to copy

Neuronpedia publishes its data as a public S3 bucket of `.jsonl.gz` files, 1
JSON object on each line, with `model.jsonl`, `source.jsonl` and `release.jsonl`
beside the data ([exports docs][exports]). The lenses are on Hugging Face under
Apache 2.0, as `torch.save` dictionaries with a `config.yaml` beside each file
that "records the source, the fitting configuration, and the command that wrote
the file" ([Hugging Face][hf-lens]). Our release should look the same: data
files, a manifest beside them, and the command that wrote them.

## 3. Where the data can live

| Host | Limits (2026-11) | Fit for us |
|------|------------------|------------|
| GitHub repository | A file over 100 MiB is blocked; a file over 50 MiB warns; "ideally less than 1 GB, and less than 5 GB is strongly recommended" ([about large files][gh-large]) | Code, manifests, checksums, a sample of 10 runs. Not the full runs |
| GitHub Releases | "We don't limit the total size of the binary files in the release or the bandwidth used to deliver them"; each file under the LFS limit of the plan, 2 GB on Free ([about large files][gh-large]) | The tar files of the runs and the store parts. No DOI |
| Git LFS | 10 GiB storage and 10 GiB bandwidth for each month on Free; "Git LFS cannot be used with GitHub Pages sites" ([LFS billing][gh-lfs], [LFS][gh-lfs-about]) | No. The bandwidth stops after about 20 downloads |
| GitHub Pages | A site of 1 GB; no LFS | The static index and the run files (476 MB) fit, but the webapp does not: it needs a server and a database |
| Zenodo | 50 GB for each record, 100 files; a DOI; "dividing a single large dataset into many records ... is not considered fair usage" ([Zenodo][zenodo]) | The archival copy. Tar the runs and the store parts into fewer than 100 files |
| Hugging Face datasets | "There is no per-repo size limit for models and datasets"; public storage is "Best-effort"; a file up to 500 GB ([HF storage][hf-storage]) | The mirror that the live viewer reads. HF serves files with CORS for a browser |

## 4. The recommendation

### 4.1 The GitHub repository: the code and the proof

One public repository, for example `urban-aesthetics-lens`. It holds:

| Part | Content |
|------|---------|
| `neuronpedia/` | A git submodule at `e2573d3`, plus `patches/*.patch` for the 2 local changes (the column label "Contrast"; the 2 model ids). No fork of the full history is necessary |
| `viewer/` | `docker-compose.yml` with Postgres 16 + pgvector and the webapp; `load_shares.py` (from `neuronpedia_load.py`); a `Makefile` with `make up` that seeds the rows and starts the viewer with no inference host |
| `export/` | `hop_viewer.py`, `hop_store.py` and `tests/test_hop_viewer.py`, so that a reader can make the runs again from the store parts |
| `read/` | `hop_all.py`, `hop_depth.py` and `research/monocle/jlens_read.py`, so that a reader with a GPU can make the store parts again from the model and the lens |
| `analysis/` | The scripts of F10.7 to F10.24 that write the numbers of the paper, and the paperkit numbers file |
| `MANIFEST.json` | sha256 of every data file, the model ids and revisions, the lens file names on Hugging Face and their sha256, the prompt, the layers, the dtype, the seed, and the command that wrote each file |
| `verify.py` | Downloads the Zenodo record, checks every sha256, makes the runs again from the store parts, and compares them with the published runs byte for byte |
| `sample/` | 10 runs and their store rows, under 20 MB, so that the viewer works with no download |

The proof of integrity is `verify.py`. A reader runs 1 command and sees that the
published runs are what the published reads give, and that the reads carry the
hashes of the manifest. A reader with a GPU goes 1 step further and makes the
reads again from the public lens and the public model.

### 4.2 The data: Zenodo for the record, Hugging Face for the viewer

| Host | Content | Files |
|------|---------|-------|
| Zenodo (DOI) | `runs_{model}.tar.gz` (the 599 runs and `shares.json`), `store_{model}_{prompt}.tar.gz` (the 4 store parts), `patch_words_{model}.tar.gz` (optional), `MANIFEST.json` | Under 100 files, about 2.5 GB |
| Hugging Face dataset | The same files, unpacked, so that each run has a URL | About 40,000 files is too many for 1 folder: put each model in its own folder, under the 10,000 limit |

Zenodo is the citation and the frozen copy. Hugging Face is the copy that a
browser reads. The paper cites the DOI.

### 4.3 The live viewer on GitHub Pages

The user asked for GitHub Pages as the host of the live copy. Pages serves
static files only, and the stock Neuronpedia webapp is a server:

| Fact | Source |
|------|--------|
| `next.config.js` sets `output: 'standalone', // for docker`, with `redirects()` and 1 rewrite | [next.config.js][nextconf] |
| `apps/webapp/middleware.ts` exists | [webapp tree][webapp] |
| The jlens page is a server component. It calls `prisma.model.findUnique`, `prisma.jlensShare.findUnique` and `resolveHosts(...)`, then renders `<JlensPageClient modelId share inferenceAvailable />` | [page.tsx][page] |
| A Next.js static export does not support proxy (middleware), rewrites, redirects, or dynamic routes without `generateStaticParams()` | [static exports][next-export] |

Thus the full app cannot go to Pages. But the share view needs very little
from the server. The client component takes 3 props: the model id, the share
row, and `inferenceAvailable`. Its only network call is
`fetch(share.url, { cache: 'force-cache' })` ([jlens-page-client.tsx][client]).
It reads the global context for 1 function, `getInferenceEnabledForModel`, and
it does not read the session.

**The plan for Pages** is thus a thin static wrapper, not the full app:

1. A new Next.js project with `output: 'export'` in the release repository,
   with the Neuronpedia source as a submodule and `@/components/jlens/*` mapped
   to it with a `tsconfig` path.
2. One route, `/[modelId]/jlens/`, with `generateStaticParams()` for the 3
   model ids. The page is a client component. It reads `shareId` from
   `useSearchParams()`, finds the row in a bundled `shares.json`, and renders
   the share view with `inferenceAvailable={false}`.
3. A stub `GlobalProvider` that returns `false` from
   `getInferenceEnabledForModel`, and the 2 patches (the column label, the
   model ids).
4. A GitHub Actions workflow that downloads the run files from the Zenodo
   record at build time and publishes the site. The repository stays small,
   and the builds-per-hour limit does not apply to a custom workflow
   ([Pages limits][gh-pages]).

**The risk is small (verified 2026-11-02).** We read the share components
([jlens-completion.tsx][completion], [jlens-chat.tsx][chat],
[jlens-lens-mode.tsx][lens-mode], [global-provider.tsx][provider],
[lens.ts][lens-ts], [env.ts][env-ts]):

| Fact | Where |
|------|-------|
| All are `'use client'` components with no import from `@/lib/db`, `next/headers`, `next-auth` or `server-only`. The provider imports `@prisma/client` for types and enums only | all |
| No `fetch` on mount. The completion and chat components send network calls only from Steer, Swap and Send, through `runLensStream`. The provider fetches only when a feature modal opens or when `refreshGlobal`/`refreshUser` are called | completion, chat, provider |
| The provider is seeded from props (`initialModels`, ...). `getInferenceEnabledForModel` reads `globalModels[modelId].inferenceEnabled` and the env list `NEXT_PUBLIC_STEER_FORCE_ALLOW_INSTRUCT_MODELS` | provider |
| The share data enters as the prop `loadedData` (a `JlensExportCompletion`), with `sharedDescription`, `sharedAttribution` and `inferenceAvailable` | completion, chat |
| With `inferenceAvailable={false}`, Send, Steer and the assistant edits show an "Inference unavailable" dialog, as on our instance | completion, chat |
| `jppLensAvailable(modelId)` reads `NEXT_PUBLIC_JPP_LENS_MODEL_IDS`, split on commas and inlined at build time | lens-ts, env-ts |

Thus the wrapper is: a `GlobalProvider` seeded with the 3 `Model` rows
(`inferenceEnabled: false`), the `LensColumnsContext` set to J++ and
Contrast, and `JlensCompletion` with `loadedData` from the run file,
`inferenceAvailable={false}`, and `sharedDescription` from `shares.json`. The
Tailwind config and the shadcn components come from the submodule. The only
file we did not read is `jlens-export.tsx` (404 at `main`; it exists at our
pinned commit, since the client imports `parseFixture` from it). Estimate: 1
to 2 days, mostly the build configuration and the 2 patches.

**The size.** "Published GitHub Pages sites may be no larger than 1 GB", with
"a soft bandwidth limit of 100 GB per month" ([Pages limits][gh-pages]). The
runs of the 3 models come to 765 MB, plus 14 MB of thumbnails and the
app. This fits, but near the limit. If a fourth model comes, move the run
files to the Hugging Face mirror and keep the app alone on Pages. A reader
opens a few runs, not all of them, so the bandwidth limit is far away.

The repository with `make up` gives the same viewer on a laptop, and that is
the part that proves the result. The live copy is for the reader who does not
want to install Docker.

### 4.4 The subset

| Include | Reason |
|---------|--------|
| All 599 runs of each of the 3 viewer models | 779 MB with the thumbnails; the viewer is the point |
| The 591 thumbnails | 14 MB; licensed for distribution (section 5) |
| The 4 store parts of every prompt variant, 3 models | 1.8 GB; every number of F10 comes from these |
| The patch top words | 530 MB for each model; optional, and the viewer does not read them |
| The lens sha256 and the Hugging Face file name | The lens is public already |

| Leave out | Reason |
|-----------|--------|
| The patch residuals | 47 GB for each model. Write "available on request" in the README |
| The lens files | Neuronpedia hosts them. A second copy adds nothing |
| The gate and the SLURM scripts | Site-specific |

## 5. Decisions (2026-11-02)

1. **The thumbnails.** The Cyclomedia imagery is licensed for distribution
   under an agreement with NYC DOB (the user, 2026-11-02). The 591 thumbnails
   go in the release.
2. **Qwen3.6-27B.** The viewer now holds 3 models. The store for
   `qwen3.6-27b` uses the J++ lens `outputs/_monocle/jlens/jpp/qwen3.6-27b/lens.pt`,
   16 layers (0 to 60, step 4), 591 images. We added the model to `MODELS`
   and `NAMES` in `hop_viewer.py`, to `MODELS` in `neuronpedia_load.py`
   (63 layers, width 5120, owner Qwen), and to `NEXT_PUBLIC_JPP_LENS_MODEL_IDS`
   in `neuronpedia_up.sh`. The layer count follows the convention of the 2
   other rows: the hidden layer count minus 1.
3. **The host of the live copy.** GitHub Pages, with the thin wrapper of
   section 4.3. Vercel with a free Postgres is the fallback.

## 6. The order of work

1. Write `MANIFEST.json` from the store and the lens files (sha256 takes about
   1 hour for the 9 GB of lenses).
2. Make the patches from the 2 `sed` lines of `neuronpedia_up.sh`.
3. Write `docker-compose.yml` and `load_shares.py`; test `make up` on a laptop.
4. Write `verify.py`; run it on the local data; it must pass before the upload.
5. Upload to Zenodo and Hugging Face; put the DOI in the README and the paper.
6. Build the Pages wrapper of section 4.3; if it does not build in 2 days,
   deploy the full app on Vercel with a free Postgres.

## 7. The review period: anonymous and not public (2026-11-02)

The user set 2 more constraints: the link must carry no author identity, and
the material is for a few reviewers, not the public. These rule out most of
the hosts of section 3 during review:

| Host | Problem during review | Source |
|------|----------------------|--------|
| GitHub Pages | On Free, "GitHub Pages in public repositories" only. A private (access-controlled) site needs an organization on GitHub Enterprise Cloud | [plans][gh-plans], [visibility][gh-vis] |
| Vercel password | "Available on Pro for $20 per month per protected project ... Not available on Hobby" | [Vercel][vercel-pw] |
| Hugging Face gated or private | "To download files from a gated dataset you'll need to be authenticated." A reviewer has no account we can grant | [HF gated][hf-gated] |
| Anonymous GitHub | Removes names and logins from a repository, but "By default, Anonymous Github and anonymous.4open.science access public repositories." It gives an unlisted link, not a gate | [README][anon-gh] |
| Zenodo open record | Public, with our names | |

Two hosts fit:

| Host | What it gives | Source |
|------|---------------|--------|
| Cloudflare Pages, Free | A static site with up to 20,000 files, 25 MiB for each file, and Pages Functions. A `functions/_middleware.js` of 20 lines checks HTTP Basic Auth against an environment variable. The URL is `<name>.pages.dev` with a name we choose. Direct upload with `wrangler` needs no GitHub link | [limits][cf-limits] |
| Zenodo restricted record | Files that the public cannot download, a secret link that "provides access to your record" and "can also have an expiry date", with permission "Can view" for "restricted/embargoed files" | [link sharing][zenodo-links] |

### 7.1 The plan

| Part | Host during review | Content |
|------|--------------------|---------|
| The viewer | Cloudflare Pages behind Basic Auth. The password goes in the supplementary material | The static wrapper of section 4.3, the 3 models' runs (765 MB, 1,797 files, each under 1 MB), the 591 thumbnails, `index.html` |
| The code | The same site, as `code.zip`, and inside the Zenodo record | The release repository of section 4.1, with the scrub of section 7.2 |
| The proof | The same site | `MANIFEST.json`, `verify.py`, and `README.md` with the chain of custody |
| The large data | Zenodo, restricted, creator "Anonymous", title with the submission number | `store_*.tar.gz` (1.8 GB), `runs_*.tar.gz`, `patch_words_*.tar.gz` (optional) |

The reviewer gets 2 links and 1 password: the viewer site with its password,
and the Zenodo secret link. `verify.py` takes the Zenodo secret link as its
argument, so the check works during review and after.

### 7.2 The scrub

The viewer export is clean. A grep for our logins, names, institution,
cluster paths and host names over the 1,797 run files, `shares.json` and
`index.html` found only the vocabulary word "unicorn" in the lens readouts.
The store manifests are clean. The 4 scripts of `research/scripts/f10/` that hold
`/share/ju/matt/zoo/models/...` paths must point at the Hugging Face model ids
in the release. The release archive must also omit `neuronpedia_gate.py`, the
SLURM scripts, `server.env`, and the git history. Run the same grep over the
archive before the upload.

### 7.3 After acceptance

Set the Zenodo record to open and add the authors; the files and the DOI do
not change (we did not verify that Zenodo keeps the DOI across this change; if
it does not, make a new version). Move the site to GitHub Pages under the
public repository, or keep Cloudflare Pages and drop the middleware. Make the
GitHub repository public. The paper cites the DOI.

### 7.4 Not verified

- The Zenodo pages we read do not say whether a restricted record gets a DOI
  at publication and keeps it when the access changes. We believe it does.
  Check in the Zenodo deposit form before the upload.
- The Cloudflare Free plan's monthly bandwidth for Pages is not on the limits
  page. A reviewer who opens 50 runs moves about 25 MB, so this does not
  matter during review.

## 8. The static share site (built 2026-11-02)

The user asked for share pages of the readouts that the paper reports, with the Neuronpedia
interface, for a few reviewers. `research/scripts/f10/share_static/` builds them.

| File | Purpose |
|------|---------|
| `share_static.py` | Clones Neuronpedia at `e2573d3` into `/scratch/$USER/neuronpedia-static/src`, applies the 2 label patches, assembles a Next.js project with `output: 'export'` in `.../app` (symlinks to the webapp's `node_modules`, `components`, `lib`, `types`, `prisma`; the client files of the jlens route; the templates), copies the run files to `public/runs/`, runs `next build`, writes `MANIFEST.json` and `README.txt`, and zips the output |
| `templates/next.config.js` | Static export, no type check, `resolve.symlinks = false` so the symlinked sources compile as project files |
| `templates/layout.tsx` | The root layout: the webapp's `Providers` seeded from `data/models.json` (3 models, inference off), no navbar, no session |
| `templates/share-page.tsx` | `/{model}/jlens/{share id}/`: `generateStaticParams` from `data/shares.json`, renders `JlensPageClient` with `inferenceAvailable={false}` inside `Suspense` |
| `templates/model-page.tsx` | `/{model}/jlens/`: the run list of one model. The model tabs and "Free Chat" land here |
| `templates/index-page.tsx` | `/`: groups by model |
| `check_static.py` | Playwright: serves the output, opens every page, and compares the clicked readout with the run file. `--base-path` serves a build for a sub-path; `--base-url` checks a live site |

Three more changes at assembly: the toggle options list our 3 models with no demos (the demos
of neuronpedia.org do not exist here); the 3 hard-coded tabs of the selector (Qwen 3.6 27B,
Gemma 3 12B, DeepSeek V4 Flash) and its "Other Models" dropdown (inference-enabled models only,
so empty here) become one tab for each of our models (`patch_tabs`, 2026-11-03); and a CSS
rule hides the Tutorial button (its tour navigates to those demos). The public files `chinese-translations-qwen.json` and the icons come along,
because the share view fetches the translation table.

**The result.** 24 pages (8 groups x 3 models), 126 files, 6.5 MB zip at
`data/f10/hop_all/f10-jlens-share.zip`. The check passes on the build and on a fresh
extraction of the zip: every page loads with no console error, shows its description and
every prompt token, and a click on the first selected token opens the readout, whose top word
at each layer, for the J++ column and the Contrast column, equals the first entry of the run
file (624 of 624 layer readouts, 0 mismatches). A grep over the zip for our logins, names,
institution and host names finds only the vocabulary words "franchise" and "unicorn".

**What a reviewer does.** Unzip, run `python3 -m http.server 8000` in the folder, open
`http://localhost:8000/`. The pages fetch the run files, so a double-click on `index.html`
does not work. Steer, Swap, Send and "Load Oracle Lens" (Qwen3.6-27B) show "unavailable".

**To add runs.** `--all` builds every run of the viewer folder (1,797 pages, about 780 MB).

```
python research/scripts/f10/share_static/share_static.py            # the zip
python research/scripts/f10/share_static/share_static.py --base-path /artificial-aesthetic \
  --dest artificial-aesthetic/site/jlens-share --no-zip    # the GitHub Pages copy
PLAYWRIGHT_BROWSERS_PATH=/scratch/$USER/neuronpedia-static/pw-browsers \
  /scratch/$USER/neuronpedia-static/pwvenv/bin/python research/scripts/f10/share_static/check_static.py
```

## 9. The GitHub Pages hookup (2026-11-03)

The user asked for a GitHub Pages site in the paper repository that serves the share package.
The site is live at `https://mfranchi.net/artificial-aesthetic/`. The live check passes: 24 of
24 pages, 624 of 624 layer readouts equal to the run files.

| Part | Where | What it does |
|------|-------|--------------|
| The site folder | `artificial-aesthetic/site/jlens-share/` (127 files, 21 MB in git) | The static export, built for the path `/artificial-aesthetic`. A `.nojekyll` file is in it. Do not edit it |
| The workflow | `artificial-aesthetic/.github/workflows/pages.yml` | On a push to `main` that touches the folder, it checks the sha256 of each run file against `MANIFEST.json`, then uploads the folder with `actions/upload-pages-artifact` and deploys it with `actions/deploy-pages`. It does not build |
| The Pages setting | Repository settings, Pages | Source "GitHub Actions" (set with the API on 2026-11-03). HTTPS enforced |
| The build command | mllmsci | `python research/scripts/f10/share_static/share_static.py --base-path /artificial-aesthetic --dest artificial-aesthetic/site/jlens-share --no-zip` |

**The base path.** A project site serves under `/<repository>/`, so the build takes
`--base-path`. The flag sets `basePath` and `assetPrefix` in `next.config.js`, prefixes the run
URLs in `shares.json` (the client fetches them as absolute paths), and patches the one absolute
path in the Neuronpedia source (`TRANSLATIONS_URL` in `lib/utils/chinese-translations.ts`).
`next/link` and `router.push` add the base path by themselves. The zip for local use keeps an
empty base path, so the two outputs differ and the script builds them one at a time.

**The URL carries the name of the author.** The user site of the account has the custom domain
`mfranchi.net`, so every project site of the account serves under it, and
`pendulating.github.io/artificial-aesthetic/` redirects there. The repository is private, but a
Pages site is public. The user chose to deploy as is: the reviewers get the package through
Anonymous GitHub (anonymous.4open.science), which hides the names. Note from section 7:
Anonymous GitHub works on a public repository only, so the repository, or a copy with the site
folder, must be public for that step.

**The download for reviewers (2026-11-03).** Anonymous GitHub shows the repository, but it does
not serve the site (the pages fetch assets at `/artificial-aesthetic/...`). So the reviewer
downloads the repository as a ZIP and runs `python3 serve.py` in it. `serve.py` (repository
root, standard library only) serves `artificial-aesthetic/site/jlens-share` at the path `/artificial-aesthetic/`,
redirects `/` there, and opens the browser. The steps are the first section of the README.
The check passes on a fresh copy of the tracked files served by `serve.py` (24 of 24 pages,
624 of 624 readouts). One caution: the Anonymous GitHub term list must not match a word in
the run files (a replacement changes a readout and breaks the sha256 in `MANIFEST.json`), so
keep the terms to names and logins.

**The Anonymous GitHub website (2026-11-03).** Anonymous GitHub has a "page" option. Its
source (`src/server/routes/webview.ts`, read on 2026-11-03) serves the files of the Pages source
of the repository (branch and path from the GitHub API; for our Actions deploy the API reports
`main` and `/`) at `https://anonymous.4open.science/w/<id>/<path>`, with an index file lookup
for a folder. So no change of the Pages setting is necessary, and the site is at
`/w/<id>/site/anon/`. Three constraints come with it, and the build has an answer to each:

| Constraint of `/w/` | Effect | Answer in the build |
|---------------------|--------|---------------------|
| The URL has the prefix `/w/<id>/site/anon` | Absolute asset paths must carry it | `--base-path /w/<id>/site/anon`. The id is the repository id that the user chose on Anonymous GitHub (`mwfpa`, 2026-11-03); a new id needs a rebuild |
| Every page gets `Content-Security-Policy: sandbox allow-scripts allow-popups allow-forms allow-modals`: an opaque origin, and the files carry no CORS header | A `fetch()` of a run file fails; fonts fall back; `localStorage` throws (the webapp catches it) | `--inline-runs`: each run goes into its page as a plain-text `data:` URL (the anonymizer matches terms at word boundaries, so plain JSON is as safe as the run files; base64 would hide a chance match) |
| A client-side `router.push` fetches an RSC payload, which the sandbox blocks, and the router then lands on the `index.txt` file | The model tabs and Free Chat break | `patch_navigation`: every `router.push` of the share client becomes a full page load, and the index and model pages use plain `<a>` links with the base path, not `next/link`, in all variants |

`check_static.py --sandbox` serves a build with that header and no CORS, and the check passes
(24 of 24 pages, 624 of 624 readouts); a click-through (index link, model tab, run link, Free
Chat) passes too. The anonymized repository must be updated on Anonymous GitHub after each push.
The size of `artificial-aesthetic/site/anon` is about 40 MB (each page carries its run twice: in the HTML and in the
RSC payload).

**To update the site.** Build with the command above, run the check with
`--base-path /artificial-aesthetic`, then commit the folder and push. The workflow deploys in
about 1 minute. To check the live site: `check_static.py --out artificial-aesthetic/site/jlens-share
--base-url https://mfranchi.net/artificial-aesthetic`.

## Sources

- [schema]: https://raw.githubusercontent.com/hijohnnylin/neuronpedia/main/apps/webapp/prisma/schema.prisma
- [client]: https://raw.githubusercontent.com/hijohnnylin/neuronpedia/main/apps/webapp/app/%5BmodelId%5D/jlens/jlens-page-client.tsx
- [share-route]: https://raw.githubusercontent.com/hijohnnylin/neuronpedia/main/apps/webapp/app/api/lens/share/route.ts
- [selfhost]: https://docs.neuronpedia.org/open-source/self-hosting
- [readme]: https://github.com/hijohnnylin/neuronpedia
- [exports]: https://docs.neuronpedia.org/api-and-data/exports
- [hf-lens]: https://huggingface.co/neuronpedia/jacobian-lens
- [gh-large]: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- [gh-lfs]: https://docs.github.com/en/billing/concepts/product-billing/git-lfs
- [gh-lfs-about]: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage
- [zenodo]: https://support.zenodo.org/help/en-gb/1-upload-deposit/80-what-are-the-size-limitations-of-zenodo
- [hf-storage]: https://huggingface.co/docs/hub/storage-limits
- [page]: https://raw.githubusercontent.com/hijohnnylin/neuronpedia/main/apps/webapp/app/%5BmodelId%5D/jlens/page.tsx
- [nextconf]: https://raw.githubusercontent.com/hijohnnylin/neuronpedia/main/apps/webapp/next.config.js
- [webapp]: https://github.com/hijohnnylin/neuronpedia/tree/main/apps/webapp
- [next-export]: https://nextjs.org/docs/app/guides/static-exports
- [gh-pages]: https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits
- [gh-plans]: https://docs.github.com/en/get-started/learning-about-github/githubs-plans
- [gh-vis]: https://docs.github.com/en/pages/getting-started-with-github-pages/changing-the-visibility-of-your-github-pages-site
- [vercel-pw]: https://vercel.com/docs/deployment-protection/methods-to-protect-deployments/password-protection
- [hf-gated]: https://huggingface.co/docs/hub/datasets-gated
- [anon-gh]: https://github.com/tdurieux/anonymous_github
- [cf-limits]: https://developers.cloudflare.com/pages/platform/limits/
- [zenodo-links]: https://help.zenodo.org/docs/share/link-sharing/
- [completion]: https://raw.githubusercontent.com/hijohnnylin/neuronpedia/main/apps/webapp/components/jlens/jlens-completion.tsx
- [chat]: https://raw.githubusercontent.com/hijohnnylin/neuronpedia/main/apps/webapp/components/jlens/jlens-chat.tsx
- [lens-mode]: https://raw.githubusercontent.com/hijohnnylin/neuronpedia/main/apps/webapp/components/jlens/jlens-lens-mode.tsx
- [provider]: https://raw.githubusercontent.com/hijohnnylin/neuronpedia/main/apps/webapp/components/provider/global-provider.tsx
- [lens-ts]: https://raw.githubusercontent.com/hijohnnylin/neuronpedia/main/apps/webapp/lib/utils/lens.ts
- [env-ts]: https://raw.githubusercontent.com/hijohnnylin/neuronpedia/main/apps/webapp/lib/env.ts

Related: [neuronpedia-viewer.md](neuronpedia-viewer.md), [lens-registry.md](lens-registry.md),
[f10-judgment-mechanism.md](f10-judgment-mechanism.md).
