# SpliceAI Vector Scan

`SpliceAI_vectorScan.py` is a CLI for scanning nucleotide sequences with the 5-model **SpliceAI-10k** ensemble and exporting **per-base splice probabilities** for:

- **neither**
- **acceptor**
- **donor**

It can score the sequence as provided and, by default, also the reversed input order of that sequence (important for testing insert orientation; see the warning below that reverse ≠ reverse-complement).

This tool is aimed at researchers who want to use SpliceAI on novel or arbitrary sequences. For example:

- synthetic circuits and expression cassettes
- coding or noncoding inserts
- viral or vector sequences
- designed transcript segments
- any construct that may be exposed to the human spliceosome

It is especially useful when you want to catch cryptic splice acceptors/donors early, before spending more time iterating designs that you *do not* want to be spliced.

---

## Table of contents

- [Why this exists](#why-this-exists)
	- [What SpliceAI is](#what-spliceai-is)
	- [How this script diverges from the original use case](#how-this-script-diverges-from-the-original-use-case)
- [Important warning: `reverse` does **not** mean reverse-complement](#important-warning-reverse-does-not-mean-reverse-complement)
	- [Why not reverse-complement?](#why-not-reverse-complement)
- [Installation](#installation)
	- [Environment expectations](#environment-expectations)
	    - [Non-stdlib Python packages](#non-stdlib-python-packages)
		- [Minimal install](#minimal-install)
		- [Recommended full install](#recommended-full-install)
- [Model files, ensembling, and caching](#model-files-ensembling-and-caching)
	- [Why this script ensembles 5 models](#why-this-script-ensembles-5-models)
	- [Why there is a cached `.keras` ensemble wrapper](#why-there-is-a-cached-keras-ensemble-wrapper)
	- [Where the cache goes](#where-the-cache-goes)
- [Model licensing](#model-licensing)
- [Quick start](#quick-start)
	- [Basic FASTA scan](#basic-fasta-scan)
	- [Use an explicit model directory](#use-an-explicit-model-directory)
	- [Generate both summary CSV and per-model member CSV](#generate-both-summary-csv-and-per-model-member-csv)
	- [Highest reproducibility-oriented run](#highest-reproducibility-oriented-run)
	- [Directly feed a sequence from the terminal](#directly-feed-a-sequence-from-the-terminal)
	- [Multiple sequences via stdin with blank-line separation](#multiple-sequences-via-stdin-with-blank-line-separation)
- [CLI usage](#cli-usage)
	- [Core options](#core-options)
	- [Context caveat](#context-caveat)
- [Input formats](#input-formats)
- [Plain-text fallback parser](#plain-text-fallback-parser)
- [Output formats](#output-formats)
- [Output naming and duplicate-sequence behavior](#output-naming-and-duplicate-sequence-behavior)
- [Logging and determinism](#logging-and-determinism)
- [Recommended invocation patterns](#recommended-invocation-patterns)
- [Interpreting the outputs](#interpreting-the-outputs)
	- [Mean probabilities vs geometric mean](#mean-probabilities-vs-geometric-mean)
	- [conf_margin_entropy is not a true confidence estimate](#conf_margin_entropy-is-not-a-true-confidence-estimate)
- [Bin labels and why these thresholds were chosen](#bin-labels-and-why-these-thresholds-were-chosen)
- [Manifest and reproducibility](#manifest-and-reproducibility)
- [Practical caveats](#practical-caveats)
- [Troubleshooting](#troubleshooting)
	- [“Failed to import TensorFlow”](#failed-to-import-tensorflow)
	- [“Failed to import Keras”](#failed-to-import-keras)
	- [“Model file not found”](#model-file-not-found)
	- [“Invalid characters in sequence”](#invalid-characters-in-sequence)
	- [CRAM fails to open](#cram-fails-to-open)
	- [bigWig fails](#bigwig-fails)
	- [OOM during inference or track writing](#oom-during-inference-or-track-writing)
- [Citation](#citation)
- [Contributing / project status](#contributing--project-status)

---

## Why this exists

### What SpliceAI is

SpliceAI (Jaganathan *et al.*, 2019) is a deep residual neural network that predicts, from primary pre-mRNA sequence alone, whether each position is a splice acceptor, donor, or neither.

Key background from the original work:

- It was trained on human pre-mRNA transcript sequence with splice annotations derived from GENCODE.
- The published SpliceAI-10k model uses 10,000 nt of sequence context total (effectively ±5 kb flank around each predicted position).
- The authors trained 5 independent models and averaged them at inference time.
- Its original intended use was chiefly human splice-junction prediction and variant interpretation, especially detecting cryptic splice-altering variants outside the canonical GT/AG dinucleotides.

The original paper also showed that the model generalized beyond standard protein-coding contexts; notably, it retained strong performance on lincRNAs, which supported the idea that it was learning properties of spliceosome recognition rather than only coding-gene idiosyncrasies.

### How this script diverges from the original use case

This script does **not** implement the classic “reference vs alternate allele Δ-score around a human locus” workflow.

Instead, it directly scores arbitrary sequences you give it.

That is a deliberate stretch beyond the original training setting, but it is a useful one. In practice, these models have already proven appreciably informative in broader contexts, including noncanonical sequence design problems and even systems such as the HIV-1 viral genome. That makes it worth exploring them more broadly in other systems exposed to the human spliceosome.

For synthetic biology and engineering biology, this is often exactly the practical question:

> “If I build or insert this sequence in human cells, where are the places the spliceosome may want to act?”

That can save substantial time in iterative design of constructs you do not want to be spliced.

This script was written to make those models easier to use from the command line by a broader scientific community. It is still in **active development**, so issues, feature requests, bug reports, and PRs are very welcome.

---

## Important warning: `reverse` does **not** mean reverse-complement

By default, this script scores:

1. the sequence in the input order (`sense`)
2. the reversed input order (`reverse`)

That second pass is useful if you are asking things like:

- “If I flip this insert, do I accidentally create spliceable structure?”
- “Is the same cassette safe in both orientations?”

### Why not reverse-complement?

Because the models were trained on human pre-mRNA sequence in the forward orientation relevant to splice recognition. Although reverse-complementing may feel theoretically symmetric in some settings, that is not the distribution the models were trained on. Using reverse-complemented inputs would therefore produce outputs with unpredictable behavior.

Also note:

- In CSV output, reverse-mode predictions are mapped back onto the original left-to-right coordinates, so you can compare `sense` and `reverse` row-by-row.
- In BED output, `+` means sense scan and `-` means reverse-order scan, **not** genomic strand.

---

## Installation

## Environment expectations

Use a normal modern Python 3 environment that can install `TensorFlow`. A fresh virtualenv or Conda environment is strongly recommended.

### Non-stdlib Python packages

The following packages do not come with stock Python 3, but are utilized by this script:

| Package        |               Required? | When needed                                                                   | Why                                                                                                                |
| -------------- | ----------------------: | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `numpy`        |                     Yes | Always                                                                        | Array handling and output processing                                                                               |
| `tensorflow`   | Yes, recommended ≥1.2.0 | Always                                                                        | Runs model inference                                                                                               |
| `keras`        |     Usually recommended | Always, unless your TensorFlow install provides compatible `tf.keras` support | Model loading / ensemble wrapper                                                                                   |
| `h5py`         | Recommended in practice | When loading legacy SpliceAI `.h5` weights                                    | Should already be included as dependency of Keras/tf.keras<br>Note this package is not directly imported by script |
| `loguru`       |                Optional | If you want prettier logs, debug messages, or better traceback formatting     | Improved CLI logging and error display                                                                             |
| `biopython`    |                Optional | FASTA / FASTQ / GenBank / EMBL input                                          | Structured sequence parsing                                                                                        |
| `pysam`        |                Optional | SAM / BAM / CRAM / VCF / BCF input                                            | Alignment and variant-file parsing                                                                                 |
| `pyBigWig`     |                Optional | `--out-format bigwig`                                                         | bigWig writing                                                                                                     |
| `platformdirs` |                Optional | For nicer persistent ensemble cache location                                  | Otherwise the script falls back to a temp directory                                                                |
| `spliceai`     |                Optional | Only if you want the script to auto-find bundled SpliceAI model files         | Alternative to providing `--model-dir`                                                                             |

### Important note about the `spliceai` package

You do not need the `spliceai` Python package if you already have the five model files:

- `spliceai1.h5`
- `spliceai2.h5`
- `spliceai3.h5`
- `spliceai4.h5`
- `spliceai5.h5`

In that case, just point the script at them with:

```bash
--model-dir /path/to/model_dir
```

If you *do* install `spliceai`, this script only uses it as a convenient place to locate the model files. It does **not** rely on the package's API/dependencies for inference logic.

### Minimal install

If you will provide your own model directory, and are looking to use as few packages as possible:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy tensorflow
```

### Recommended full install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy tensorflow keras h5py spliceai loguru biopython pysam pyBigWig platformdirs
```

---

## Model files, ensembling, and caching

### Why this script ensembles 5 models

That is not an arbitrary design choice here, as it mirrors the original SpliceAI inference procedure.

In the paper, the authors trained 5 separate models and used the average of their outputs during testing. This script preserves that published behavior.

### Why there is a cached `.keras` ensemble wrapper

The original SpliceAI weights are distributed as legacy `.h5` model files. Modern Keras runtimes prefer the `.keras` format for native model serialization. This script therefore:

1. loads the five original SpliceAI members
2. wraps them into a single multi-output ensemble model
3. caches that wrapper as `spliceai_ensemble.keras`

This does **not** retrain, alter, or reinterpret the learned weights. It is purely a compatibility/performance convenience layer around the original ensemble.

### Where the cache goes

- If `--model-dir` is provided, the cached wrapper is written there.
- Otherwise it is placed in a user cache directory (if `platformdirs` is installed) or a temp directory fallback.

If the cached ensemble ever becomes incompatible after environment changes, deleting it and rerunning is usually sufficient.

---

## Model licensing

From the README of the maintained fork of the SpliceAI package:

> "The trained models used by SpliceAI (located in this package at spliceai/models) are provided under the [CC BY NC 4.0](https://github.com/bw2/SpliceAI/blob/master/LICENSE) license for academic and non-commercial use; other use requires a commercial license from Illumina, Inc."

Please make sure your use of the model weights is consistent with that license.

---

## Quick start

### Basic FASTA scan

```bash
python3 SpliceAI_vectorScan.py \
  -i constructs.fa \
  -o results/constructs
```

### Use an explicit model directory

```bash
python3 SpliceAI_vectorScan.py \
  -i constructs.fa \
  -o results/constructs \
  --model-dir /path/to/spliceai_models
```

### Generate both summary CSV and per-model member CSV

`csv-members` should be treated as a companion output to `csv`, not a replacement. The downstream (WIP) graphing interface expects them together.

```bash
python3 SpliceAI_vectorScan.py \
  -i constructs.fa \
  -o results/constructs \
  --out-format csv csv-members
```

### Highest reproducibility-oriented run

```bash
python3 SpliceAI_vectorScan.py \
  -i constructs.fa \
  -o results/constructs \
  --cpu \
  --determinism hi \
  --metadata yes-hash
```

### Directly feed a sequence from the terminal

```bash
printf 'AUGGUAAGUCCAGGUAAGUCCAGGUAAGU\n' | \
python3 SpliceAI_vectorScan.py -i - -o results/stdin_demo
```

### Multiple sequences via stdin with blank-line separation

```bash
cat <<'EOF' | python3 SpliceAI_vectorScan.py -i - -o results/stdin_multi
AUGGUAAGUCCAGGUAAGU

GGGCUCUCUUACAGGUAAGUCUCAG
EOF
```

---

## CLI usage

```bash
python3 SpliceAI_vectorScan.py -i INPUT -o OUT_PREFIX [options]
```

### Core options

| Option                                              | Default                                             | Meaning                                                           |
| --------------------------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------- |
| `-h, --help`                                        | N/A                                                 | Display CLI arguments                                             |
| `-i, --in-file`                                     | required                                            | Input file path, or `-` for stdin                                 |
| `-o, --out-prefix`                                  | required                                            | Output path/prefix, without extension                             |
| `--model-dir PATH`                                  | auto-discover from installed `spliceai` if possible | Directory containing `spliceai1.h5` … `spliceai5.h5`              |
| `--out-format ...`                                  | `csv`                                               | One or more of: `csv`, `csv-members`, `bedgraph`, `bed`, `bigwig` |
| `--gzip`                                            | off                                                 | Gzip text outputs (`csv`, `csv-members`, `bedgraph`, `bed`)       |
| `--no-reverse`                                      | off (WILL do reverse scoring)                       | Disable reversed-input-order scoring                              |
| `--context INT`                                     | `10000`                                             | Total SpliceAI context; `10000` matches SpliceAI-10k              |
| `--cpu`                                             | off                                                 | Force CPU even if GPU is available                                |
| `--determinism {no,none,lo,low,med,medium,hi,high}` | `low`                                               | Reproducibility vs throughput tradeoff                            |
| `--predict-mode {call,on_batch}`                    | `call`                                              | Inference method                                                  |
| `--metadata {no,yes,yes-hash}`                      | `no`                                                | Write a manifest JSON for reproducibility                         |
| `--version`                                         | off                                                 | Print version banner and exit                                     |

### Context caveat

`--context 10000` is the intended setting for SpliceAI-10k.

If you reduce it, the model sees less flank and predictions near sequence ends are more likely to degrade. If you increase it, you are mostly adding more padding Ns. This is unlikely to improve results.

---

## Input formats

## Accepted input file types

| Format             | Typical extensions                        | Dependency  | What gets scored                                        | Status           | Important assumptions                                                                         |
| ------------------ | ----------------------------------------- | ----------- | ------------------------------------------------------- | ---------------- | --------------------------------------------------------------------------------------------- |
| FASTA              | `.fa`, `.fasta`, `.fna`, optional `.gz`   | `biopython` | Each sequence record                                    | Stable           | Standard structured parser                                                                    |
| FASTQ              | `.fq`, `.fastq`, optional `.gz`           | `biopython` | Each read sequence                                      | Stable           | Useful for small curated sets; not intended as an RNA-seq quantifier                          |
| GenBank            | `.gb`, `.gbk`, `.genbank`, optional `.gz` | `biopython` | Each record sequence                                    | Stable           | Standard structured parser                                                                    |
| EMBL               | `.embl`, optional `.gz`                   | `biopython` | Each record sequence                                    | Stable           | Standard structured parser                                                                    |
| SAM                | `.sam`, `.sam.gz`                         | `pysam`     | Each read’s `query_sequence`                            | **Experimental** | Scores read/query sequence only; does **not** reconstruct reference context                   |
| BAM                | `.bam`                                    | `pysam`     | Each read’s `query_sequence`                            | **Experimental** | Same caveat as SAM                                                                            |
| CRAM               | `.cram`                                   | `pysam`     | Each read’s `query_sequence`                            | **Experimental** | Same caveat as SAM; CRAM may also require a reference to open                                 |
| VCF                | `.vcf`, `.vcf.gz`                         | `pysam`     | Literal `REF` and each non-symbolic `ALT` allele string | **Experimental** | Does **not** reconstruct genomic windows or haplotypes; symbolic/breakend alleles are skipped |
| BCF                | `.bcf`                                    | `pysam`     | Literal `REF` and each non-symbolic `ALT` allele string | **Experimental** | Same caveat as VCF                                                                            |
| Plain text / stdin | any text, or `-`                          | none        | Sequence lines/blocks                                   | Stable           | See fallback parser below                                                                     |

### Why SAM/BAM/CRAM and VCF/BCF are marked experimental

These formats are supported for convenience, but the interpretation is intentionally simple with them:

- **Alignment files**: the script uses the read/query sequence exactly as present in the file.
- **Variant files**: the script uses the literal REF/ALT string only.

This script does **not**:

- fetch the reference genome
- reconstruct ±5 kb native genomic context
- assemble haplotypes
- reproduce the classic SpliceAI variant-delta workflow, (that's what the original package and it's successors are for)

---

## Plain-text fallback parser

If the input is not handled as a structured biological file, the script falls back to a simple text parser.

### Behavior

- If the text contains blank lines, each blank-line-delimited block is treated as one sequence.
- Otherwise, each non-empty non-header line is treated as one sequence.
- Lines beginning with `>`, `@`, `#`, or `;` are ignored.
- Whitespace and digits are removed.
- Sequence is uppercased before scoring.

### Accepted alphabet

The model encoder accepts:

- `A`, `C`, `G`, `T`
- `U` (treated like `T`)
- IUPAC ambiguity codes: `N, R, Y, S, W, K, M, B, D, H, V`

Ambiguity codes are treated as N-like unknown bases during one-hot encoding because the models were not trained on their specific semantics.
Any other character triggers a clear error with traceback.

---

## Output formats

## Supported output file types

| Format        | Files produced                                                                | Dependency | Coordinates                       | Contents                                                           | Memory profile                |
| ------------- | ----------------------------------------------------------------------------- | ---------- | --------------------------------- | ------------------------------------------------------------------ | ----------------------------- |
| `csv`         | `<prefix>.<record>.(sense\|reverse).csv[.gz]`                                 | none       | 1-based per base                  | Mean probabilities, geometric-mean scores, hybrid confidence, bins | Streamed; usually memory-safe |
| `csv-members` | `<prefix>.<record>.(sense\|reverse).members.csv[.gz]`                         | none       | 1-based per base                  | Per-model probabilities for all 5 ensemble members                 | Streamed; usually memory-safe |
| `bedgraph`    | `<prefix>.<record>.(sense\|reverse).(neither\|acceptor\|donor).bedgraph[.gz]` | none       | 0-based half-open                 | Dense 1-bp signal track                                            | Streamed; usually memory-safe |
| `bed`         | `<prefix>.spliceai_bins.bed[.gz]`                                             | none       | 0-based half-open                 | Run-length-compressed acceptor/donor bins only                     | Can OOM on large jobs         |
| `bigwig`      | `<prefix>.(sense\|reverse).(acceptor\|donor).mean.bw`                         | `pyBigWig` | Browser-style indexed dense track | Mean acceptor/donor tracks only                                    | Can OOM on large jobs         |
| `gff3`        | WIP                                                                           | WIP        |                                   |                                                                    |                               |

### Notes by format

#### `csv`
Per-base summary output. Columns are:

- `record_id`,  `pos_1based`, `strand`, `base`,
- `p_neither_mean`, `p_acceptor_mean`, `p_donor_mean`,
- `gmean_score_neither`, `gmean_score_acceptor`, `gmean_score_donor`,
- `conf_margin_entropy`, `bin_acceptor`, `bin_donor`

#### `csv-members`
Per-base, per-model output. This is intended as a companion to `csv` to be used with a (WIP) graphing interface.

#### `bedgraph`
Dense signal, one file per record × strand × class. Good if you want every base retained in a text browser-track format.

#### `bed`
Single aggregated BED6 file of bin-compressed intervals for acceptor and donor only. Important details:

- It excludes bins below `0.05`:
	- `not_a_site`
	- `unlikely_site`
- Score is scaled to `0–1000`.
- `name` is prefixed with `ACC:` or `DON:`.
- `strand` is `+` for the sense scan and **`-` for the reverse-order scan**.

#### `bigwig`
Writes dense indexed binary tracks for:

- sense acceptor
- sense donor
- reverse acceptor
- reverse donor

This does **not** write a `neither` bigWig.

### Why `bed` and `bigwig` can run out of memory (OOM)

Compared with CSV and bedGraph, these two formats are less streaming-friendly in this implementation:

- **BED**: intervals are collected in memory and globally sorted before writing.
- **bigWig**: dense arrays are retained until final emission in raw input order.

So for many long sequences, `bed` and `bigwig` are more likely to OOM, whereas `csv`, `csv-members`, & `bedgraph` are much less likely to, because they are written more incrementally (as each sequence is scanned).

### Compression note

`--gzip` applies to text outputs:

- `csv`
- `csv-members`
- `bedgraph`
- `bed`

It does **not** apply to `bigwig`.

---

## Output naming and duplicate-sequence behavior

The script deduplicates exactly identical sequences for inference, which can save time.

### Important consequence

- `csv` and `csv-members` are produced per unique sequence
- BED-family outputs are emitted per raw input record

If duplicates were collapsed for inference, CSV record IDs may be merged with `|`.

This is useful for designed construct panels, but if you feed very large read-level files (FASTQ/BAM/SAM), be aware that the output semantics are not the same as a read-counting pipeline.

Also:

- filenames are sanitized for filesystem safety
- BED/BEDGRAPH/bigWig chromosome names are sanitized to BED-compatible identifiers and made unique

---

## Logging and determinism

## Logging (`LOGLEVEL` environment variable)

Set `LOGLEVEL` before running the script.

| `LOGLEVEL` | Accepted aliases | Meaning |
|---|---|---|
| `DEBUG` | `TRACE` | Most verbose; useful for debugging |
| `INFO` | — | Default; general run information |
| `WARNING` | `WARN` | Warnings and errors only |
| `ERROR` | `ERR` | Errors only |
| `CRITICAL` | `CRIT`, `FATAL` | Critical failures only |

Example:

```bash
LOGLEVEL=DEBUG python3 SpliceAI_vectorScan.py -i constructs.fa -o results/constructs
```

If you install **`loguru`**, you will get:

- prettier CLI logging
- easier-to-read debug output
- better traceback formatting

Without `loguru`, the script falls back to standard-library logging.

## Determinism (`--determinism`)

| Setting  | Aliases | What it does                                                                                           | Typical use                                      |
| -------- | ------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| `none`   | `no`    | No determinism-related TensorFlow settings                                                             | Maximum throughput, moderate reproducibility     |
| `low`    | `lo`    | Enables deterministic ops where practical, without strongest thread restrictions                       | **Recommended default** for almost all use cases |
| `medium` | `med`   | Adds single-thread intra/inter-op settings                                                             | More reproducible, slower                        |
| `high`   | `hi`    | Strongest determinism settings available in this script, including extra TF/cuDNN determinism controls | Best effort for strict reproducibility           |

### Important reproducibility note

For absolute determinism, you should also use `--cpu`. Even then, exact reproducibility still depends on holding the full stack constant:

- Python version
- NumPy / TensorFlow / Keras versions
- model files
- hardware / drivers

In practice `--determinism lo` is sufficient for almost all routine design work; it is much more performant than forcing maximal determinism.

---

## Recommended invocation patterns

### 1. Standard design-screening run

```bash
python3 SpliceAI_vectorScan.py \
  -i designs.fa \
  -o out/designs \
  --out-format csv bed
```

### 2. Design-screening run with per-model details for downstream plotting

```bash
python3 SpliceAI_vectorScan.py \
  -i designs.fa \
  -o out/designs \
  --out-format csv csv-members
```

### 3. Faster run when you do not care about reversed orientation

```bash
python3 SpliceAI_vectorScan.py \
  -i designs.fa \
  -o out/designs \
  --no-reverse
```

### 4. All browser tracks for inspection

```bash
python3 SpliceAI_vectorScan.py \
  -i designs.fa \
  -o tracks/designs \
  --out-format bedgraph bed bigwig
```

### 5. Reproducibility-first run with manifest hashing

```bash
python3 SpliceAI_vectorScan.py \
  -i designs.fa \
  -o out/designs \
  --cpu \
  --determinism hi \
  --metadata yes-hash
```

### 6. Use model files you downloaded yourself

```bash
python3 SpliceAI_vectorScan.py \
  -i designs.fa \
  -o out/designs \
  --model-dir /path/to/spliceai_h5_models
```

---

## Interpreting the outputs

## Mean probabilities vs geometric mean

### `p_*_mean`
These are the straightforward **arithmetic means** of the five SpliceAI model outputs.

For each base, these are the closest thing to the ensemble’s main prediction, and they should behave like probabilities across:

- neither
- acceptor
- donor

(i.e. they sum to approximately 1, aside from tiny floating-point effects).

### `gmean_score_*`
These are **per-class geometric means across ensemble members**.

They are intentionally not renormalized.

That means:

- `gmean_score_neither`
- `gmean_score_acceptor`
- `gmean_score_donor`

will not necessarily sum to 1 at a base, which is expected behaviour.

### Why include geometric mean at all?

Because it exposes ensemble disagreement that the arithmetic mean can hide.

If one or two models strongly dissent while the others agree, the ordinary mean can still look moderately confident. The geometric mean penalizes that disagreement much more strongly, so it works as an “at a glance” indicator of whether the ensemble is truly aligned.

So:

- use `p_*_mean` as the main probability-like output
- use `gmean_score_*` as a disagreement-sensitive support score

---

## `conf_margin_entropy` is **not** a true confidence estimate

The `conf_margin_entropy` column is a **hybrid heuristic score**, not a calibrated confidence value.

It combines:

- the margin between the top two classes
- the entropy of the three-class mean probability vector

In code, it is essentially:

$$ 
C = \frac{2}{3}(p_1 - p_2) + \frac{1}{3}\left(1 - \frac{H(p)}{\log 3}\right)
$$

where $p_1$ and $p_2$ are the top two normalized class probabilities and $H(p)$ is entropy.

### What it is for

It is there to help with quick visual triage:

- “Is this base sharply peaked into one class?”
- “Does this row look diffuse/ambiguous?”

### What it is **not**:

- a statistical confidence interval
- a calibrated posterior probability
- a Bayesian credible measure
- a substitute for rigorous uncertainty analysis

A more rigorous uncertainty/statistical layer belongs in the separate (WIP) graphing interface. Here the goal is simply an at-a-glance summary.

---

## Bin labels and why these thresholds were chosen

The script bins **mean acceptor** and **mean donor** probabilities into the following labels:

| Range | Label |
|---|---|
| `[0.0, 0.001)` | `not_a_site` |
| `[0.001, 0.05)` | `unlikely_site` |
| `[0.05, 0.2)` | `insignificant_site` |
| `[0.2, 0.35)` | `weak_cryptic_tissue_specific` |
| `[0.35, 0.5)` | `significant_cryptic_tissue_specific` |
| `[0.5, 0.8)` | `significant` |
| `[0.8, 1.0]` | `very_strong_site` |

### Why these labels are biologically motivated

These bins are **heuristic interpretive labels** for this CLI. They are not official SpliceAI class names. The thresholds were chosen to line up with empirical regimes discussed in the original SpliceAI paper:

- Very low scores are usually background-like.
- Around **0.2**, the paper used permissive thresholds that already enriched for real splice-altering events.
- The range **0.35–0.8** was especially interesting because the paper found many weak/intermediate cryptic events in this regime to be associated with alternative and tissue-specific splicing.
- Around **0.5**, predicted cryptic splice variants became substantially more convincing.
- Above **0.8**, events were the strongest, most deleterious, and less likely to be merely weak context-dependent competition.

So the label names are intended to convey the following intuition:

- **below 0.05**: probably negligible for most practical screening
- **0.2–0.5**: weak-to-moderate cryptic potential; often the regime where context and tissue dependence matter
- **0.5–0.8**: strong enough to care about immediately
- **>0.8**: very strong site-like behavior

### Important caveat

The original paper’s best-validated thresholds were mostly discussed in the context of variant Δ-scores, not arbitrary-sequence direct scanning. So these labels are useful heuristics, not hard clinical categories.

---

## Manifest and reproducibility

Use:

```bash
--metadata yes
```

or, preferably:

```bash
--metadata yes-hash
```

which writes to:

```text
<out_prefix>.manifest.json
```

### What is in the manifest

The manifest records the parts of the run that matter for reproducibility, including:

- tool name and version
- Python / NumPy / TensorFlow / Keras versions
- context length
- predict mode
- whether reverse scoring was enabled
- device selection
	- CPU vs GPU
	- whether a GPU was available
	- whether GPU visibility was disabled
- determinism level
- TensorFlow determinism settings actually applied
- base model paths
- cached ensemble path
- whether the ensemble was loaded from cache
- optional SHA256 hash of the ensemble model representation/weights
	- At time of writing, the ensemble hashes to "`2c4bf868b41296074ba0c35589778adec9744ec6e5328e22b86d9d5d88438e5b`"
- raw vs unique input record counts

### Why this matters

Neural-network inference is not reproducible from sequence alone. Results can differ if you change:

- model files
- Keras/TensorFlow versions
- CPU vs GPU
- determinism settings
- cached wrapper state
- context length

The manifest gives you a concrete provenance record, which is essential if you want to:

- compare runs over time
- rerun a design screen later
- share results with collaborators
- debug subtle output differences
- support publication-quality reproducibility

If you care strongly about reproducibility, use:

```bash
--metadata yes-hash --cpu --determinism hi
```

---

## Practical caveats

- This is full-length inference. Very long sequences can OOM at inference time even before output writing.
- `--no-reverse` can roughly halve the inference burden if you do not need orientation testing.
- `bed` and `bigwig` are the most memory-demanding output modes in this implementation.
- The model was trained for human spliceosome-relevant sequence contexts; using it far outside that domain is exploratory.
- The script is excellent for screening and prioritization, but it is not a substitute for direct biological validation when decisions are expensive.

---

## Troubleshooting

### “Failed to import TensorFlow”
Install TensorFlow in the active environment.

### “Failed to import Keras”
Install `keras`, or use a TensorFlow build with compatible `tf.keras`.

### “Model file not found”
Either:

- install `spliceai` so the models can be discovered automatically, or
- provide `--model-dir` containing `spliceai1.h5` … `spliceai5.h5`

### “Invalid characters in sequence”
Allowed characters are:

- `A C G T U`
- `N R Y S W K M B D H V`

Anything else is rejected.

### CRAM fails to open
`pysam`/CRAM may need access to a reference. Configure the appropriate reference path or cache for your environment.

### bigWig fails
Install `pyBigWig`.

### OOM during inference or track writing
Try one or more of:

- shorter sequences
- `--no-reverse`
- `--cpu`
- avoiding `bed` and `bigwig`
- writing only `csv` / `bedgraph`

---

## Citation

If you use these model outputs scientifically, please cite the original SpliceAI paper:

**Jaganathan K, Kyriazopoulou Panagiotopoulou S, McRae JF, et al.**  
*Predicting Splicing from Primary Sequence with Deep Learning.*  
**Cell** (2019).  
DOI: [10.1016/j.cell.2018.12.015](https://doi.org/10.1016/j.cell.2018.12.015)

---

## Contributing / project status

### This tool is in **active development**.

Please open:
- issues
- feature requests
- bug reports
- pull requests

If something is unclear, breaks on your environment, or would make the tool more useful for real research workflows, feel free to reach out to me through GitHub or [this email forwarder](mailto:6s1j4fwkg@mozmail.com).

---
