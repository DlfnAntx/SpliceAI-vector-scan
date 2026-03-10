import argparse
import csv
import gzip
import io
import sys
import re
from typing import Iterable, List, Tuple, Union, IO, Optional, Dict, Tuple as _Tuple
import numpy as np
import os
import json
import hashlib

ENSEMBLE_BASENAME = "spliceai_ensemble"

# Logging
# - Prefer loguru if available (nicer CLI output).
# - Fall back to stdlib logging with a tiny shim that supports logger.add/remove.
import logging


def _get_logger():
    """Return (logger, has_loguru) with a minimal loguru-like shim if needed."""
    try:
        from loguru import logger as _loguru_logger  # type: ignore
        return _loguru_logger, True
    except Exception:
        base = logging.getLogger("spliceai_vector_scan")

        class _Shim:
            """Minimal loguru-like surface used by this script."""
            def __init__(self, lg: logging.Logger):
                self._lg = lg
                self._handlers = []

            def remove(self):
                for h in list(self._handlers):
                    try:
                        self._lg.removeHandler(h)
                    except Exception:
                        pass
                self._handlers = []

            def add(self, sink, level="INFO", backtrace=False, diagnose=False, format=None):
                if hasattr(sink, "write"):
                    h = logging.StreamHandler(sink)
                else:
                    h = logging.FileHandler(str(sink))
                fmt = format or "%(levelname)-7s | %(message)s"
                h.setFormatter(logging.Formatter(fmt))
                self._lg.setLevel(getattr(logging, str(level).upper(), logging.INFO))
                self._lg.addHandler(h)
                self._lg.propagate = False
                self._handlers.append(h)
                return h

            def debug(self, msg, *a, **k): self._lg.debug(msg, *a, **k)
            def info(self, msg, *a, **k): self._lg.info(msg, *a, **k)
            def warning(self, msg, *a, **k): self._lg.warning(msg, *a, **k)
            def error(self, msg, *a, **k): self._lg.error(msg, *a, **k)

            def exception(self, msg, *a, **k):
                """Log an exception with traceback information."""
                self._lg.exception(msg, *a, **k)

            def log(self, level, msg, *a, **k):
                """Log a message with a numeric or string severity."""
                if isinstance(level, str):
                    lvl = getattr(logging, level.upper(), logging.INFO)
                else:
                    lvl = level
                self._lg.log(lvl, msg, *a, **k)

        return _Shim(base), False


logger, _HAS_LOGURU = _get_logger()

__version__ = "1.6.2"


def _normalize_log_level(level: Optional[str]) -> _Tuple[str, Optional[str]]:
    """Normalize a user-provided log level. Input: raw level string; output: (normalized level, warning message)."""
    if level is None:
        return "INFO", None
    s = str(level).strip().upper()
    if not s:
        return "INFO", None
    alias = {"WARN": "WARNING", "ERR": "ERROR", "FATAL": "CRITICAL", "CRIT": "CRITICAL", "TRACE": "DEBUG"}
    s = alias.get(s, s)
    valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if s not in valid:
        return "INFO", f"Unrecognized LOGLEVEL={level!r}; defaulting to INFO."
    return s, None


def _configure_logging(log_level_env: Optional[str]) -> str:
    """Configure logger sinks. Input: log level string; output: normalized level used."""
    log_level, warn = _normalize_log_level(log_level_env)
    logger.remove()
    if _HAS_LOGURU:
        logger.add(sys.stderr, level=log_level, backtrace=False, diagnose=False,
                   format="<level>{level: <7}</level> | {message}")
    else:
        logger.add(sys.stderr, level=log_level,
                   format="%(levelname)-7s | %(message)s")
        if hasattr(logger, "_lg"):
            logger._lg.propagate = False
    if warn:
        logger.warning(warn)
    return log_level


def _log_debug_exception(message: str, exc: BaseException) -> None:
    """Log an exception with traceback at DEBUG level. Input: message and exception; output: None."""
    if _HAS_LOGURU:
        # Use loguru's opt to include the exception traceback with debug
        try:
            logger.opt(exception=exc).debug(message)
        except Exception:
            # fallback to plain debug
            logger.debug(message)
    else:
        try:
            exc_info = (type(exc), exc, exc.__traceback__)
            logger.debug(message, exc_info=exc_info)
        except Exception:
            logger.debug(message)


# TensorFlow/Keras are imported lazily to avoid import-time side effects (env vars, device init, etc.).
tf = None
load_model = None
Input = None
Model = None
layers = None
clone_model = None


def _normalize_determinism_level(level: str) -> str:
    """Normalize user-facing determinism aliases to canonical levels."""
    s = (level or "low").strip().lower()
    m = {
        "no": "none", "none": "none",
        "lo": "low", "low": "low",
        "med": "medium", "medium": "medium",
        "hi": "high", "high": "high",
    }
    if s not in m:
        raise ValueError(f"Invalid determinism level: {level!r}")
    return m[s]


def _configure_tf_env(effective_log_level: str, determinism_level: str) -> str:
    """Set TF env vars *before* importing tensorflow."""
    lvl = (effective_log_level or "INFO").upper()
    tf_log_map = {
        "DEBUG": "0",
        "INFO": "1",
        "WARNING": "2",
        "ERROR": "2",
        "CRITICAL": "3",
    }
    tf_log = tf_log_map.get(lvl, "1")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = tf_log

    det = _normalize_determinism_level(determinism_level)

    if det == "none":
        os.environ.pop("TF_DETERMINISTIC_OPS", None)
        os.environ.pop("TF_CUDNN_DETERMINISTIC", None)
    else:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        if det == "high":
            os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
        else:
            os.environ.pop("TF_CUDNN_DETERMINISTIC", None)

    return det


def _apply_tf_determinism(det: str) -> Dict[str, object]:
    """Apply TF runtime settings after import (best-effort)."""
    applied: Dict[str, object] = {"level": det}

    if det in ("medium", "high"):
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            applied["intra_op_threads"] = 1
            applied["inter_op_threads"] = 1
        except Exception:
            pass

    if det == "high":
        try:
            enable = getattr(getattr(tf.config, "experimental", None), "enable_op_determinism", None)
            if callable(enable):
                enable()
                applied["op_determinism_enabled"] = True
        except Exception:
            pass

    return applied


def _suppress_benign_tf_warnings() -> None:
    """Suppress known benign warnings emitted during model load."""
    try:
        import logging as _logging
        _logging.getLogger("absl").addFilter(
            lambda r: "No training configuration found in the save file" not in r.getMessage()
        )
    except Exception:
        pass


def _ensure_tf_keras() -> None:
    """Import tensorflow/keras on demand and bind symbols used in the module."""
    global tf, load_model, Input, Model, layers, clone_model
    if tf is not None and load_model is not None:
        return

    try:
        import tensorflow as _tf
    except Exception as e:
        raise RuntimeError("Failed to import TensorFlow. Please ensure it is installed.") from e

    # Prefer standalone Keras 3, fallback to tf.keras
    try:
        from keras.saving import load_model as _load_model
        from keras import Input as _Input, Model as _Model, layers as _layers
        from keras.models import clone_model as _clone_model
    except Exception:
        try:
            from tensorflow.keras.models import load_model as _load_model  # type: ignore
            from tensorflow.keras import Input as _Input, Model as _Model, layers as _layers  # type: ignore
            from tensorflow.keras.models import clone_model as _clone_model  # type: ignore
        except Exception as e2:
            raise RuntimeError("Failed to import Keras from either standalone or tf.keras.") from e2

    tf = _tf
    load_model = _load_model
    Input = _Input
    Model = _Model
    layers = _layers
    clone_model = _clone_model

    _suppress_benign_tf_warnings()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and enforce required inputs."""
    p = argparse.ArgumentParser(
        description="Scan cassette (pre-mRNA) sequences with SpliceAI (input order + reversed input order) and export per-base probabilities + bins."
    )
    p.add_argument(
        "-i", "--in-file",
        dest="in_file",
        help="Input file containing nucleotide sequences "
             "(FASTA/FASTQ/GenBank/EMBL/SAM/BAM/CRAM/VCF/BCF or plain text). "
             "Note: SAM/BAM/CRAM provide query/read sequences; VCF/BCF provide literal allele strings; "
             "this tool does not reconstruct reference windows from coordinates. "
             "Or use '-' for stdin."
    )
    p.add_argument(
        "-o", "--out-prefix",
        dest="out_prefix",
        help="Output prefix/path (no extension)."
    )
    p.add_argument(
        "--determinism",
        choices=["no", "none", "lo", "low", "med", "medium", "hi", "high"],
        default="low",
        help="Determinism vs performance tradeoff: no|none, lo|low, med|medium, hi|high (default: low)."
    )
    p.add_argument(
        "--metadata",
        choices=["no", "yes", "yes-hash"],
        default="no",
        help="Write <out_prefix>.manifest.json with run metadata: no|yes|yes-hash. In yes-hash, include a SHA256 over the ensemble model (cached .keras file when available). Can fall back to hasing the weights if the serialization is unavailable."
    )
    p.add_argument(
        "--context", type=int, default=10000,
        help="Total context S (padding length); use 10000 for SpliceAI-10k (default)."
    )
    p.add_argument(
        "--model-dir", type=str, default=None,
        help="Directory containing spliceai1.h5..spliceai5.h5. If omitted, will default to SpliceAI's package resources."
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available (GPU visibility disabled).")
    p.add_argument("--no-reverse", dest="no_reverse", action="store_true", help="Disable reversed-sequence scoring (reverse order, not reverse-complement).")
    p.add_argument(
        "--predict-mode",
        choices=["call", "on_batch"],
        default="call",
        help="Use direct call (fast path) or predict_on_batch for inference (default: call)."
    )
    p.add_argument(
        "--out-format",
        dest="out_format",
        nargs="+",
        choices=["csv", "csv-members", "bedgraph", "bed", "bigwig"],
        default=["csv"],
        help="One or more output formats to write (default: csv). Example: --out-format csv csv-members bed bigwig"
    )
    p.add_argument("--gzip", action="store_true", help="Gzip-compress outputs (.gz).")
    p.add_argument("--version", action="store_true", help="Print version-like info and exit.")
    args = p.parse_args()

    if not args.version:
        if args.in_file is None or args.out_prefix is None:
            p.error("-i/--in-file and -o/--out-prefix are required unless --version is specified.")

    return args


def version_banner() -> None:
    """Print a lightweight version banner without importing TF/Keras."""
    import platform as _platform
    banner = (
        f"spliceai_vector_scan.py v{__version__}\n"
        " - SpliceAI ensemble: 10k-context (5 models) wrapped into one cached .keras ensemble wrapper\n"
        " - Extracts nucleotide sequence(s) from many file types (FASTA/FASTQ/GenBank/EMBL/SAM/BAM/CRAM/VCF/BCF or plain text)\n"
        " - Device: auto GPU/CPU (override with --cpu)\n"
        " - Full-length per-sequence inference\n"
        " - Per-base probabilities + bins; reversed input-order optional (not reverse complement)\n"
        f" - Python: {_platform.python_version()}\n"
    )
    print(banner)


def auto_device(force_cpu: bool = False, gpus: Optional[List[object]] = None) -> Tuple[str, bool]:
    """Return (device string, gpu_available) using provided GPU list or TF detection."""
    if gpus is None:
        gpus = _detect_gpus()
    gpu_available = bool(gpus)
    if force_cpu or not gpu_available:
        return "/CPU:0", gpu_available
    return "/GPU:0", True


def _detect_gpus() -> List[object]:
    """Return a list of TF-discovered physical GPUs (best-effort, built-in)."""
    try:
        return tf.config.list_physical_devices("GPU")
    except Exception:
        return []


def _disable_gpus() -> bool:
    """Attempt to disable visible GPUs in TF; returns True if successful."""
    try:
        tf.config.set_visible_devices([], "GPU")
        return True
    except Exception:
        return False


BIN_LABELS = [
    "not_a_site",                           # [0.0, 0.001)
    "unlikely_site",                        # [0.001, 0.05)
    "insignificant_site",                   # [0.05, 0.2)
    "weak_cryptic_tissue_specific",         # [0.2, 0.35)
    "significant_cryptic_tissue_specific",  # [0.35, 0.5)
    "significant",                          # [0.5, 0.8)
    "very_strong_site",                     # [0.8, 1.0]
]

# Output filename hygiene
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_record_id(raw: str, max_len: int = 128) -> str:
    """Make record IDs safe for filenames across platforms."""
    s = (raw or "").strip()
    s = _SAFE_ID_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    if not s:
        s = "record"
    if len(s) > max_len:
        s = s[:max_len]
    return s


# BED v1 compliance utilities (hts-specs BEDv1, 2022)
# - chrom must match [[:alnum:]_]{1,255} (i.e. [A-Za-z0-9_]{1,255})
_BED_CHROM_BAD = re.compile(r"[^A-Za-z0-9_]+")


def sanitize_bed_chrom(raw: str, max_len: int = 255) -> str:
    """Sanitize an arbitrary record label into a BEDv1-compliant chrom field."""
    s = (raw or "").strip()
    s = _BED_CHROM_BAD.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "record"
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
        if not s:
            s = "record"
    return s


def make_unique_bed_chroms(raw_names: List[str]) -> List[str]:
    """Generate BEDv1-compliant chrom names from raw record labels, ensuring uniqueness."""
    out: List[str] = []
    seen: set = set()
    for i, nm in enumerate(raw_names, 1):
        base = sanitize_bed_chrom(nm)
        chrom = base
        if chrom in seen:
            suffix = f"_{i}"
            keep = 255 - len(suffix)
            if keep < 1:
                chrom = ("r" * max(1, keep)) + suffix
            else:
                chrom = base[:keep].rstrip("_") + suffix
        if chrom in seen:
            k = 2
            while True:
                suffix = f"_{i}_{k}"
                keep = 255 - len(suffix)
                chrom2 = base[:max(1, keep)].rstrip("_") + suffix
                if chrom2 not in seen:
                    chrom = chrom2
                    break
                k += 1
        seen.add(chrom)
        out.append(chrom)
    return out


def make_unique_file_ids(raw_names: List[str], max_len: int = 128) -> List[str]:
    """Make filesystem-safe IDs from raw names (for output filenames), ensuring uniqueness."""
    out: List[str] = []
    seen: Dict[str, int] = {}
    for nm in raw_names:
        base = sanitize_record_id(nm, max_len=max_len)
        if base in seen:
            seen[base] += 1
            rid = f"{base}.{seen[base]}"
        else:
            seen[base] = 1
            rid = base
        out.append(rid)
    return out


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Return SHA256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_ensemble_model(ensemble, ensemble_path: str) -> str:
    """Hash the ensemble model for provenance."""
    if ensemble_path and os.path.exists(ensemble_path):
        return _sha256_file(ensemble_path)

    h = hashlib.sha256()
    for arr in ensemble.get_weights():
        a = np.asarray(arr)
        h.update(str(a.shape).encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def write_manifest(out_path: str, manifest: Dict[str, object]) -> None:
    """Write JSON manifest to disk."""
    logger.info(f"Writing manifest: {out_path}")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")


def _bin_indices(prob_1d: np.ndarray) -> np.ndarray:
    """Vectorized mapping of probability -> BIN_LABELS index."""
    thr = np.array([0.001, 0.05, 0.2, 0.35, 0.5, 0.8], dtype=np.float32)
    x = np.asarray(prob_1d, dtype=np.float32)
    return np.searchsorted(thr, x, side="right").astype(np.int32)


def label_bins(prob_array: np.ndarray) -> List[str]:
    """Map probabilities to bin labels."""
    idx = _bin_indices(prob_array)
    return [BIN_LABELS[int(i)] for i in idx]


#### Start Section: Input handling ####
def _open_text(path_or_file: Union[str, IO], mode: str = "rt") -> IO:
    """Open as text; .gz supported. Never double-wrap a text handle. Never close stdin."""
    if isinstance(path_or_file, io.TextIOBase):
        logger.debug("Using existing text handle")
        return path_or_file
    if isinstance(path_or_file, str):
        if path_or_file == "-":
            logger.info("Reading sequences from STDIN")
            return io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
        if path_or_file.lower().endswith(".gz"):
            logger.debug(f"Opening gzip-compressed text file: {path_or_file}")
            return io.TextIOWrapper(gzip.open(path_or_file, mode="rb"), encoding="utf-8", errors="replace")
        logger.debug(f"Opening text file: {path_or_file}")
        return open(path_or_file, mode=mode, encoding="utf-8", errors="replace")
    logger.debug("Wrapping binary handle as text")
    return io.TextIOWrapper(path_or_file, encoding="utf-8", errors="replace")


def _guess_biopy_format_from_ext(path: str) -> Optional[str]:
    """Return Biopython format string inferred from file extension."""
    lower = path.lower()
    ext_map = {
        ".fa": "fasta", ".fasta": "fasta", ".fna": "fasta",
        ".fq": "fastq", ".fastq": "fastq",
        ".gb": "genbank", ".gbk": "genbank", ".genbank": "genbank",
        ".embl": "embl",
    }
    for ext, fmt in ext_map.items():
        if lower.endswith(ext) or lower.endswith(ext + ".gz"):
            return fmt
    return None


def read_sequences(path_or_file: Union[str, IO]) -> Iterable[Tuple[str, str]]:
    """
    Yield (label, SEQUENCE_UPPERCASED) from multiple formats.

    Supports FASTA/FASTQ/GenBank/EMBL (Biopython), SAM/BAM/CRAM and VCF/BCF (pysam),
    or plain text as a fallback.
    """
    if isinstance(path_or_file, str):
        lower = path_or_file.lower()
        guess_fmt = _guess_biopy_format_from_ext(path_or_file)

        ## SAM/BAM/CRAM (+ .sam.gz) ##
        if lower.endswith((".sam", ".sam.gz", ".bam", ".cram")):
            logger.info(f"Detected alignment file format: {os.path.basename(path_or_file)}")
            try:
                import pysam
            except ModuleNotFoundError:
                raise RuntimeError("pysam is required to read SAM/BAM/CRAM")
            try:
                if lower.endswith(".cram"):
                    af = pysam.AlignmentFile(path_or_file, "rc")
                elif lower.endswith(".bam"):
                    af = pysam.AlignmentFile(path_or_file, "rb")
                elif lower.endswith(".sam.gz"):
                    af = pysam.AlignmentFile(gzip.open(path_or_file, "rt"), "r")
                else:
                    af = pysam.AlignmentFile(path_or_file, "r")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to open {path_or_file} with pysam: {e}"
                    + (" (CRAM may require a reference via reference_filename= or REF_PATH/REF_CACHE)." if lower.endswith(".cram") else "")
                )
            n = 0
            try:
                for aln in af.fetch(until_eof=True):
                    seq = getattr(aln, "query_sequence", None)
                    if not seq:
                        continue
                    qname = aln.query_name or f"read{n+1}"
                    try:
                        rname = getattr(aln, "reference_name", None)
                        rstart = getattr(aln, "reference_start", None)
                        if rname is not None and rstart is not None and rstart >= 0:
                            label = f"{qname}|{rname}:{int(rstart)+1}"
                        else:
                            label = qname
                    except Exception:
                        label = qname
                    yield (label, seq.upper())
                    n += 1
            finally:
                af.close()
                logger.info(f"Read {n} sequences from alignment file")
            return

        ## VCF/BCF (.vcf, .vcf.gz, .bcf) ##
        if lower.endswith((".vcf", ".vcf.gz", ".bcf")):
            logger.info(f"Detected variant file format: {os.path.basename(path_or_file)}")
            try:
                import pysam
            except ModuleNotFoundError:
                raise RuntimeError("pysam is required to read VCF/BCF")
            try:
                vf = pysam.VariantFile(path_or_file)
            except Exception as e:
                raise RuntimeError(f"Failed to open {path_or_file} with pysam: {e}")
            n_ref = n_alt = n_symbolic = 0
            try:
                for rec in vf:
                    chrom = getattr(rec, "chrom", None)
                    pos = getattr(rec, "pos", None)
                    rid = getattr(rec, "id", None)
                    rid = None if (rid is None or rid == ".") else str(rid)

                    ref = (rec.ref or "").upper()
                    if ref:
                        label_ref = f"{chrom}:{pos}:{rid}:REF" if rid else f"{chrom}:{pos}:REF"
                        yield (label_ref, ref)
                        n_ref += 1

                    for i, alt in enumerate(rec.alts or []):
                        a = (alt or "").strip().upper()
                        if (not a) or (a in (".", "*")) or a.startswith("<") or any(c in a for c in "[]"):
                            n_symbolic += 1
                            continue
                        label_alt = f"{chrom}:{pos}:{rid}:ALT{i+1}" if rid else f"{chrom}:{pos}:ALT{i+1}"
                        yield (label_alt, a)
                        n_alt += 1

            finally:
                vf.close()
                logger.info(
                    f"Read {n_ref} REF and {n_alt} ALT sequences from VCF/BCF "
                    f"(skipped {n_symbolic} symbolic/breakend)."
                )
            return

        if guess_fmt and "Bio" not in sys.modules:
            try:
                import Bio  # noqa: F401
            except ModuleNotFoundError:
                raise RuntimeError(
                    f"Biopython is required to read {guess_fmt.upper()} (inferred from file extension). "
                    "Install Biopython or provide a plain-text input."
                )

    ## Text-based formats (Biopython) + fallback ##
    text = _open_text(path_or_file, mode="rt")
    should_close = isinstance(path_or_file, str) and path_or_file not in ("-",)

    buffered: Optional[str] = None
    try:
        if not text.seekable():
            logger.debug("Input not seekable; buffering to memory")
            buffered = text.read()
            if should_close:
                try:
                    text.close()
                except Exception:
                    pass
            text = io.StringIO(buffered)
    except Exception:
        pass

    biopy_formats = ("fasta", "fastq", "genbank", "embl")
    try:
        from Bio import SeqIO
        for fmt in biopy_formats:
            if buffered is not None:
                handle = io.StringIO(buffered)
            else:
                try:
                    text.seek(0)
                except Exception:
                    pass
                handle = text
            got_one = False
            try:
                for rec in SeqIO.parse(handle, fmt):
                    seq = str(rec.seq).upper()
                    if not seq:
                        continue
                    label = str(rec.id) if getattr(rec, "id", None) else (str(rec.name) if getattr(rec, "name", None) else "record")
                    yield (label, seq)
                    got_one = True
                if got_one:
                    logger.info(f"Detected {fmt.upper()} and parsed sequence records")
                    if buffered is None and should_close:
                        try:
                            text.close()
                        except Exception:
                            pass
                    return
            except Exception as e:
                if isinstance(path_or_file, str) and _guess_biopy_format_from_ext(path_or_file) == fmt:
                    raise RuntimeError(f"Failed to parse {fmt.upper()} input: {e}")
                logger.debug(f"Not a valid {fmt.upper()} file")
                continue
    except ModuleNotFoundError:
        if isinstance(path_or_file, str) and _guess_biopy_format_from_ext(path_or_file):
            raise RuntimeError(
                f"Biopython is required to read {_guess_biopy_format_from_ext(path_or_file).upper()} (inferred from file extension)."
            )
        logger.debug("Biopython not installed; skipping FASTA/FASTQ/GenBank/EMBL parsing")

    ## Plain-text fallback ##
    if buffered is None:
        try:
            text.seek(0)
        except Exception:
            pass
        all_text = text.read()
        if should_close:
            try:
                text.close()
            except Exception:
                pass
    else:
        all_text = buffered

    lines = all_text.splitlines()
    has_blank = any(len(l.strip()) == 0 for l in lines)

    def _clean_line(s: str) -> str:
        return re.sub(r"[\s0-9]+", "", s).upper()

    if has_blank:
        logger.info("Parsing plain text with blank-line delimited blocks")
        i = 1
        block = []
        for raw in lines:
            if len(raw.strip()) == 0:
                if block:
                    seq = _clean_line("".join(block))
                    if seq:
                        yield (f"seq{i}", seq)
                        i += 1
                    block = []
                continue
            if raw.startswith((">", "@", "#", ";")):
                continue
            block.append(raw)
        if block:
            seq = _clean_line("".join(block))
            if seq:
                yield (f"seq{i}", seq)
    else:
        logger.info("Parsing plain text: one sequence per non-empty, non-header line")
        i = 1
        for raw in lines:
            s = raw.strip()
            if not s or s.startswith((">", "@", "#", ";")):
                continue
            seq = _clean_line(s)
            if seq:
                yield (f"seq{i}", seq)
                i += 1
#### End Section: Input handling ####


def rev(seq: str) -> str:
    """Return the sequence in reverse order (NOT reverse-complement)."""
    return seq[::-1]


def _resolve_model_paths(model_dir: Optional[str]) -> List[str]:
    """Resolve paths to spliceai1.h5..spliceai5.h5 from model_dir or package resources."""
    paths: List[str] = []
    if model_dir:
        logger.info(f"Resolving model paths from directory: {model_dir}")
        for i in range(1, 6):
            paths.append(os.path.join(model_dir, f"spliceai{i}.h5"))
        return paths

    # 1) Try find_spec to avoid importing the package.
    try:
        from importlib import util
        spec = util.find_spec("spliceai")
        if spec and spec.submodule_search_locations:
            pkg_path = spec.submodule_search_locations[0]
            models_dir = os.path.join(pkg_path, "models")
            for i in range(1, 6):
                paths.append(os.path.join(models_dir, f"spliceai{i}.h5"))
            logger.debug("Resolved model paths from installed 'spliceai' package directory (via find_spec)")
            return paths
    except Exception:

        # 2) Fallback: importlib.resources (may import spliceai)
        try:
            from importlib import resources
            for i in range(1, 6):
                paths.append(str(resources.files("spliceai") / "models" / f"spliceai{i}.h5"))
            logger.debug("Resolved model paths from installed 'spliceai' package resources")
            return paths
        except Exception as e:
            raise RuntimeError(
                "Could not resolve SpliceAI model paths from package resources. "
                "Either install 'spliceai' and 'importlib', or provide --model-dir."
            ) from e


def load_spliceai_models(paths: List[str]):
    """Load the 5-model SpliceAI ensemble from .h5 files."""
    models = []
    for pth in paths:
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Model file not found: {pth}")
        logger.debug(f"Loading model: {pth}")
        m = load_model(pth, compile=False)
        try:
            m.make_predict_function(force=True)
        except Exception:
            logger.warning("Failed to prebuild predict function.")
        models.append(m)
    logger.info(f"Loaded {len(models)} SpliceAI models")
    return models


def load_or_build_ensemble(model_dir: Optional[str]):
    """
    Load cached ensemble model if available; otherwise build from base models and cache.

    Returns:
      ensemble: Keras Model(onehot -> list of per-member outputs)
      meta: dict with resolved paths and cache info
    """
    import tempfile

    if model_dir:
        ensemble_path = os.path.join(model_dir, ENSEMBLE_BASENAME + ".keras")
    else:
        try:
            from platformdirs import user_cache_dir
            cache_root = user_cache_dir("spliceai_vector_scan", "spliceai_vector_scan")
        except Exception:
            cache_root = os.path.join(tempfile.gettempdir(), "spliceai_vector_scan_cache")
        ensemble_path = os.path.join(cache_root, ENSEMBLE_BASENAME + ".keras")

    base_paths = []
    try:
        base_paths = _resolve_model_paths(model_dir)
    except Exception:
        base_paths = []

    meta: Dict[str, object] = {
        "ensemble_path": ensemble_path,
        "loaded_from_cache": False,
        "base_model_paths": base_paths,
    }

    if os.path.exists(ensemble_path):
        logger.info(f"Loading cached ensemble model from: {ensemble_path}")
        try:
            ensemble = load_model(ensemble_path, compile=False)
            try:
                ensemble.make_predict_function(force=True)
            except Exception:
                logger.warning("Failed to prebuild predict function.")
            meta["loaded_from_cache"] = True
            return ensemble, meta
        except Exception as e:
            logger.warning(
                f"Failed to load cached ensemble at {ensemble_path}; "
                f"rebuilding from base models. Error: {e}"
            )
            _log_debug_exception("Cached ensemble load failed", e)

    logger.info("No usable cached ensemble model found; building from base SpliceAI models.")
    paths = _resolve_model_paths(model_dir)
    meta["base_model_paths"] = paths
    models = load_spliceai_models(paths)
    ensemble = build_ensemble_model(models)

    try:
        cache_dir = os.path.dirname(ensemble_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        #tmp_path = ensemble_path + ".tmp"
        ensemble.save(ensemble_path)
        #os.replace(tmp_path, ensemble_path)
        logger.info(f"Saved ensemble model to: {ensemble_path}")
    except Exception as e:
        logger.warning(
            f"Could not cache ensemble model to {ensemble_path}; continuing without cache."
        )
        _log_debug_exception("Ensemble cache write failed", e)

    try:
        ensemble.make_predict_function(force=True)
    except Exception:
        logger.warning("Failed to prebuild predict function.")

    return ensemble, meta


def build_ensemble_model(models):
    """
    Wrap N SpliceAI models into a single multi-output Keras model.

    Inputs:
      one-hot sequence: [B, L+S, 4]

    Outputs:
      list of length N, where each element is [B, L, 3] from a single model.
    """
    if not models:
        raise ValueError("build_ensemble_model(): no models provided.")
    n_models = len(models)

    logger.info(f"Building ensemble wrapper over {n_models} SpliceAI model(s)")

    # Sanity check
    try:
        _ = models[0].get_weights()
    except Exception as e:
        raise ValueError(
            "build_ensemble_model(): first element is not a valid "
            "Keras model (no get_weights())."
        ) from e
    if len(models[0].get_weights()) == 0:
        raise ValueError(
            "build_ensemble_model(): base model has no weights; "
            "unexpected SpliceAI .h5 format."
        )

    def _clone_with_prefix(m, prefix: str):
        def _renamer(layer):
            cfg = layer.get_config()
            name = cfg.get("name", layer.name)
            cfg["name"] = f"{prefix}{name}"
            return layer.__class__.from_config(cfg)

        logger.debug(f"Cloning submodel with prefix '{prefix}'.")
        cm = clone_model(m, clone_function=_renamer)
        # Force a unique **Model/Operation** name so Keras 3 doesn't collide on 'model_1'
        cm = Model(cm.inputs, cm.outputs, name=f"{prefix}member")
        cm.set_weights(m.get_weights())
        cm.trainable = False
        return cm

    # Shared symbolic input: dynamic length, 4 channels (A/C/G/T one-hot)
    inp = Input(shape=(None, 4), name="onehot")

    outs = []
    for i, m in enumerate(models, 1):
        prefix = f"m{i}_"
        cm = _clone_with_prefix(m, prefix)
        y = cm(inp, training=False)
        if isinstance(y, (list, tuple)):
            y = y[0]
        logger.debug(f"Submodel #{i} normalized output shape: {tuple(y.shape)}")
        outs.append(y)

    model = Model(inp, outs, name=ENSEMBLE_BASENAME)
    logger.info(
        "Ensemble wrapper model constructed successfully "
        "(unique submodel op names; per-model outputs only)."
    )
    return model


def one_hot_encode_with_pad(seq: str, context: int) -> np.ndarray:
    """
    Pad with N*(S/2) on both ends and one-hot encode.
    Input will be uppercase, but other invalid characters will be flagged.
    ALL IUPAC ambiguous characters are treated as N because the models would otherwise have unpredictable behavior.
    """
    lut = getattr(one_hot_encode_with_pad, "_lut", None)
    valid = getattr(one_hot_encode_with_pad, "_valid", None)
    if lut is None or valid is None:
        logger.debug("Initializing one-hot LUT")
        lut = np.zeros((256, 4), dtype=np.float32)
        lut[ord("A")] = [1.0, 0.0, 0.0, 0.0]
        lut[ord("C")] = [0.0, 1.0, 0.0, 0.0]
        lut[ord("G")] = [0.0, 0.0, 1.0, 0.0]
        for ch in b"TU":
            lut[ch] = [0.0, 0.0, 0.0, 1.0]
        for ch in b"NRYSWKMBDHV":
            lut[ch] = [0.0, 0.0, 0.0, 0.0]
        valid = np.zeros(256, dtype=bool)
        for ch in b"ACGTUNRYSWKMBDHV":
            valid[ch] = True
        one_hot_encode_with_pad._lut = lut
        one_hot_encode_with_pad._valid = valid

    ## Bad input handling ##
    b = seq.encode("ascii", errors="strict")
    idx = np.frombuffer(b, dtype=np.uint8)
    is_valid = valid[idx]
    if not np.all(is_valid):
        bad_pos = np.nonzero(~is_valid)[0]
        bad_bytes = idx[bad_pos]
        bad_chars = sorted({chr(int(x)) for x in bad_bytes})
        samples = []
        for p in bad_pos[:10]:
            lo = max(0, p - 5)
            hi = min(len(seq), p + 6)
            frag = seq[lo:hi]
            caret = " " * (p - lo) + "^"
            samples.append(f"pos {p}: {frag}\n         {caret}")
        sample_block = "\n".join(samples)
        allowed = "A,C,G,T|U and IUPAC N,R,Y,S,W,K,M,B,D,H,V"
        logger.error("Invalid characters encountered in sequence")
        raise ValueError(
            "Invalid characters in sequence.\n"
            f"  Allowed: {allowed}\n"
            f"  Found chars: {bad_chars}\n"
            f"  First bad positions (with local context):\n{sample_block}"
        )
    
    ## Pad & return ##
    core = lut[idx]
    half = context // 2
    out = np.zeros((1, core.shape[0] + context, 4), dtype=np.float32)
    if core.shape[0]:
        out[0, half:half + core.shape[0], :] = core
    return out


def _run_ensemble_one_strand(
    seq: str,
    header: str,
    strand: str,
    ensemble_model,
    context: int,
    predict_mode: str,
    need_members: bool,
    need_gmean: bool,
) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    Run ensemble on one sequence/strand.

    Returns:
      members: [L, N, 3] float32 (or None if need_members=False and need_gmean=False)
      avg:     [L, 3] float32
      gmean:   [L, 3] float32 (or None if need_gmean=False)
    """
    L = len(seq)
    x = one_hot_encode_with_pad(seq, context)  # [1, L+S, 4]
    logger.debug(f"Forward pass on full-length {strand} sequence; len={L}, context={context}")

    try:
        out = ensemble_model.predict_on_batch(x) if predict_mode == "on_batch" else ensemble_model(x, training=False)
    except tf.errors.ResourceExhaustedError:
        logger.error(
            f"OOM while running full-length inference for "
            f"record={header} strand={strand} (length={L}, context={context}). "
            "You may need to shorten your sequences or switch to a device with more memory."
        )
        raise

    if not isinstance(out, (list, tuple)):
        out = [out]
    if not out:
        raise ValueError(f"Ensemble model returned no outputs for record={header} strand={strand}.")

    per_model: List[np.ndarray] = []
    for idx, y in enumerate(out):
        if isinstance(y, (list, tuple)):
            y = y[0]

        if hasattr(y, "numpy"):
            arr = y.numpy()
        else:
            arr = np.asarray(y)

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] != L:
            raise ValueError(
                f"Submodel #{idx+1} output has shape {arr.shape}; expected [L, 3] with L={L} "
                f"for record={header} strand={strand}."
            )
        per_model.append(arr.astype(np.float32, copy=False))

    if (not need_members) and (not need_gmean):
        avg = (np.add.reduce(per_model) / float(len(per_model))).astype(np.float32, copy=False)
        return None, avg, None

    members = np.stack(per_model, axis=1)  # [L, N, 3]
    avg = members.mean(axis=1)

    gmean = None
    if need_gmean:
        eps = np.finfo(np.float32).tiny
        gmean = np.exp(np.mean(np.log(np.clip(members, eps, 1.0)), axis=1)).astype(np.float32, copy=False)

    if not need_members:
        members = None

    return members, avg.astype(np.float32, copy=False), gmean


def _hybrid_confidence_row(probs_row: np.ndarray, w: float = 2.0/3.0) -> float:
    """
    Hybrid margin+entropy confidence for a single base:
      C = w * (p_hat - s_hat) + (1 - w) * (1 - H(p)/log 3)
    """
    w = float(np.clip(w, 0.0, 1.0))
    p = np.asarray(probs_row, dtype=np.float64)
    s = p.sum()
    if not np.isfinite(s) or s <= 0.0:
        p = np.array([1/3, 1/3, 1/3], dtype=np.float64)
    else:
        p = p / s
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    p_sorted = np.sort(p)[::-1]
    p_hat, s_hat = p_sorted[0], p_sorted[1]
    c_margin = p_hat - s_hat
    H = -(p * np.log(p)).sum()
    c_entropy = 1.0 - (H / np.log(3.0))
    return float(w * c_margin + (1.0 - w) * c_entropy)


def write_csv(
    out_path: str,
    record_id: str,
    strand: str,
    probs_mean: np.ndarray,
    probs_gmean: np.ndarray,
    bases: str,
    gzip_out: bool,
):
    """
    CSV columns:
        record_id, pos_1based, strand, base,
        p_neither_mean, p_acceptor_mean, p_donor_mean,
        gmean_score_neither, gmean_score_acceptor, gmean_score_donor,
        conf_margin_entropy,
        bin_acceptor, bin_donor
    """
    opener = gzip.open if gzip_out else open
    logger.debug(f"Writing CSV: {out_path}")
    with opener(out_path, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = [
            "record_id", "pos_1based", "strand", "base",
            "p_neither_mean", "p_acceptor_mean", "p_donor_mean",
            "gmean_score_neither", "gmean_score_acceptor", "gmean_score_donor",
            "conf_margin_entropy",
            "bin_acceptor", "bin_donor",
        ]
        w.writerow(header)

        p_neither_m  = probs_mean[:, 0]
        p_acceptor_m = probs_mean[:, 1]
        p_donor_m    = probs_mean[:, 2]

        p_neither_g  = probs_gmean[:, 0]
        p_acceptor_g = probs_gmean[:, 1]
        p_donor_g    = probs_gmean[:, 2]

        bins_a = label_bins(p_acceptor_m)
        bins_d = label_bins(p_donor_m)
        for i in range(len(probs_mean)):
            c_overall = _hybrid_confidence_row(probs_mean[i, :])
            base = bases[i]
            w.writerow([
                record_id, i + 1, strand, base,
                f"{p_neither_m[i]:.6f}", f"{p_acceptor_m[i]:.6f}", f"{p_donor_m[i]:.6f}",
                f"{p_neither_g[i]:.6f}", f"{p_acceptor_g[i]:.6f}", f"{p_donor_g[i]:.6f}",
                f"{c_overall:.6f}",
                bins_a[i], bins_d[i],
            ])


def write_members_csv(
    out_path: str,
    record_id: str,
    strand: str,
    members: np.ndarray,
    bases: str,
    gzip_out: bool,
):
    """
    CSV columns:
      record_id, pos_1based, strand, base,
      p_neither_m1, p_acceptor_m1, p_donor_m1,
      ...,
      p_neither_mN, p_acceptor_mN, p_donor_mN
    """
    opener = gzip.open if gzip_out else open
    logger.debug(f"Writing ensemble-members CSV: {out_path}")

    if members.ndim != 3 or members.shape[2] != 3:
        raise ValueError(f"write_members_csv(): expected members with shape [L, N, 3], got {members.shape}")

    L, N, _ = members.shape

    with opener(out_path, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["record_id", "pos_1based", "strand", "base"]
        for j in range(N):
            header += [f"p_neither_m{j+1}", f"p_acceptor_m{j+1}", f"p_donor_m{j+1}"]
        w.writerow(header)

        for i in range(L):
            row = [record_id, i + 1, strand, bases[i]]
            row.extend(f"{members[i, j, k]:.6f}" for j in range(N) for k in range(3))
            w.writerow(row)


def write_bedgraph(
    out_path: str,
    chrom: str,
    probs: np.ndarray,
    class_index: int,
    gzip_out: bool,
):
    """Write bedGraph: chrom, start_0based, end, value."""
    opener = gzip.open if gzip_out else open
    logger.debug(f"Writing bedGraph: {out_path}")
    with opener(out_path, "wt", encoding="utf-8") as fh:
        vals = probs[:, class_index]
        # Chunked writes reduce Python overhead for long sequences, without changing semantics.
        step = 50000
        for start in range(0, vals.shape[0], step):
            end = min(vals.shape[0], start + step)
            block = [f"{chrom}\t{i}\t{i+1}\t{vals[i]:.6f}\n" for i in range(start, end)]
            fh.writelines(block)


def _bed_score_from_prob(p: float) -> int:
    """Convert probability to BEDv1 score integer in [0,1000]."""
    if not np.isfinite(p):
        return 0
    x = int(round(float(p) * 1000.0))
    if x < 0:
        return 0
    if x > 1000:
        return 1000
    return x



def _collect_bed6_bins(
    out_rows: List[Tuple[str, int, int, str, int, str]],
    chrom: str,
    probs: np.ndarray,
    class_index: int,
    strand: str,
    feature_prefix: str,
    min_label_index: int = 2,
) -> None:
    """
    Collect BED6 intervals (run-length compressed by bin label) for a single class track.
    - Bins with label index < min_label_index are skipped (default skips not_a_site, unlikely_site)
    """
    L = probs.shape[0]
    if L == 0:
        return

    vals = probs[:, class_index].astype(np.float32, copy=False)
    labels_idx = _bin_indices(vals)

    start = 0
    cur = int(labels_idx[0])
    cur_max = float(vals[0])

    for i in range(1, L):
        lb = int(labels_idx[i])
        v = float(vals[i])
        if lb != cur:
            if cur >= min_label_index:
                score = _bed_score_from_prob(cur_max)
                out_rows.append((chrom, start, i, f"{feature_prefix}{BIN_LABELS[cur]}", score, strand))
            start = i
            cur = lb
            cur_max = v
        else:
            if v > cur_max:
                cur_max = v

    # last run
    if cur >= min_label_index:
        score = _bed_score_from_prob(cur_max)
        out_rows.append((chrom, start, L, f"{feature_prefix}{BIN_LABELS[cur]}", score, strand))


class _BigWigTrackWriters:
    """
    Lazily-opened bigWig writers (pyBigWig) for mean acceptor/donor tracks.

    bigWig is effectively an indexed bedGraph; we write fixedStep per-base values (span=1, step=1),
    which is compact and fast for dense per-base signals.
    """
    def __init__(self, out_prefix: str, chrom_sizes: List[Tuple[str, int]]):
        """Initialize bigWig writers for an output prefix."""
        self.out_prefix = out_prefix
        self.chrom_sizes = chrom_sizes
        self._bw = {}

    def _open(self, key: str, path: str):
        """Open a bigWig file handle and register it."""
        import pyBigWig  # type: ignore
        bw = pyBigWig.open(path, "w")
        bw.addHeader(self.chrom_sizes)
        self._bw[key] = bw
        return bw

    def get(self, strand: str, cls_name: str):
        """Get (or open) a bigWig writer for a track."""
        key = f"{strand}:{cls_name}"
        if key in self._bw:
            return self._bw[key]
        path = f"{self.out_prefix}.{strand}.{cls_name}.mean.bw"
        return self._open(key, path)

    def add_dense_track(self, bw, chrom: str, values_1d: np.ndarray):
        """Add a dense fixedStep track for a single chrom."""
        bw.addEntries(chrom, 0, values=values_1d.astype(float).tolist(), span=1, step=1, validate=False)

    def close(self):
        """Close all bigWig handles."""
        for bw in self._bw.values():
            try:
                bw.close()
            except Exception:
                pass
        self._bw = {}


def run() -> None:
    """CLI entrypoint."""
    log_level = _configure_logging(os.environ.get("LOGLEVEL"))

    args = parse_args()
    if args.version:
        version_banner()
        sys.exit(0)

    if args.context < 0:
        logger.error("--context must be nonnegative.")
        sys.exit(2)
    if args.context % 2 != 0:
        logger.warning(f"--context {args.context} is not even; rounding down to {args.context - 1}.")
        args.context -= 1

    # Configure TensorFlow env vars *before* importing tensorflow.
    det_level = _configure_tf_env(log_level, args.determinism)
    logger.debug(f"Determinism requested: {args.determinism} -> {det_level}")
    _ensure_tf_keras()

    # GPU discovery and optional disabling (built-in TF detection).
    gpus = _detect_gpus()
    gpu_available = bool(gpus)
    gpu_disabled = False
    if args.cpu and gpu_available:
        gpu_disabled = _disable_gpus()
        if not gpu_disabled:
            logger.warning("Failed to disable GPU visibility; TensorFlow may still access the GPU.")

    # Apply runtime determinism settings
    det_applied = _apply_tf_determinism(det_level)
    det_applied["TF_DETERMINISTIC_OPS"] = os.environ.get("TF_DETERMINISTIC_OPS")
    det_applied["TF_CUDNN_DETERMINISTIC"] = os.environ.get("TF_CUDNN_DETERMINISTIC")
    logger.debug(f"TF determinism applied: {det_applied}")

    logger.info("Starting SpliceAI vector scan")
    logger.info(
        f"Args: in_file={args.in_file}, out_prefix={args.out_prefix}, out_format={args.out_format}, "
        f"gzip={args.gzip}, predict_mode={args.predict_mode}, "
        f"reverse={'off' if args.no_reverse else 'on (reverse order, not revcomp)'}"
    )

    if args.context != 10000:
        logger.warning(
            "SpliceAI models were trained/benchmarked with --context=10000 (±5000 bp padding/flank). "
            "This script will still report probabilities for every base in your input sequence, but using a different "
            "context changes how much flank the model sees; accuracy near the sequence ends may degrade if flanks are shorter "
            "than ~5000 bp. Larger context mostly adds more padded Ns and is unlikely to improve results."
        )

    device_str, _ = auto_device(force_cpu=args.cpu, gpus=gpus)
    device_label = "CPU" if args.cpu or not gpu_available else "GPU"
    logger.info(f"Device selected: {device_label} ({device_str})")
    do_reverse = not args.no_reverse

    out_formats = set(args.out_format)
    write_csv_flag = "csv" in out_formats
    write_csv_members_flag = "csv-members" in out_formats
    write_bg_flag = "bedgraph" in out_formats
    write_bed_flag = "bed" in out_formats
    write_bw_flag = "bigwig" in out_formats

    out_dir = os.path.dirname(args.out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)
    logger.debug(f"Output directory ready: {out_dir}")

    ensemble, ensemble_meta = load_or_build_ensemble(args.model_dir)

    logger.info("Reading input sequences…")
    try:
        records_raw = [(hdr, seq) for hdr, seq in read_sequences(args.in_file)]
    except Exception as e:
        logger.error(str(e))
        sys.exit(2)

    if not records_raw:
        logger.error(f"No nucleotide sequences found in {args.in_file}.")
        sys.exit(2)

    n_bases_raw = sum(len(s) for _, s in records_raw)
    logger.info(f"Collected {len(records_raw)} record(s) totaling {n_bases_raw:,} bp")

    # Group records by exact sequence for inference reuse (CSV uses merged headers; bed-based outputs remain per-raw-record).
    groups: List[Dict[str, object]] = []
    seq_to_g: Dict[str, int] = {}
    raw_to_group: List[int] = [0] * len(records_raw)
    for idx, (hdr, seq) in enumerate(records_raw):
        g = seq_to_g.get(seq)
        if g is None:
            g = len(groups)
            seq_to_g[seq] = g
            groups.append({"seq": seq, "headers": [hdr], "raw_indices": [idx]})
        else:
            groups[g]["headers"].append(hdr)
            groups[g]["raw_indices"].append(idx)
        raw_to_group[idx] = g

    n_unique = len(groups)
    if n_unique != len(records_raw):
        logger.info(
            f"Collapsed {len(records_raw) - n_unique} duplicate sequence(s) for inference "
            f"({n_unique} unique sequence(s) remain)."
        )

    # Precompute filename-safe IDs (for output filenames)
    merged_headers = ["|".join(g["headers"]) for g in groups]
    merged_ids_for_path = make_unique_file_ids(merged_headers)
    raw_ids_for_path = make_unique_file_ids([hdr for hdr, _ in records_raw])

    # Precompute BEDv1-compliant chrom names for raw records (for BED/bedGraph/bigWig)
    raw_bed_chroms = make_unique_bed_chroms([hdr for hdr, _ in records_raw])

    # Optional run metadata (only factors that can impact predictions).
    if args.metadata != "no":
        manifest_path = f"{args.out_prefix}.manifest.json"
        try:
            tf_ver = getattr(tf, "__version__", None)

            keras_ver = None
            try:
                import keras as _k  # type: ignore
                keras_ver = getattr(_k, "__version__", None)
            except Exception:
                try:
                    import tensorflow.keras as _tk  # type: ignore
                    keras_ver = getattr(_tk, "__version__", None)
                except Exception:
                    pass

            models_section: Dict[str, object] = {
                "ensemble_path": str(ensemble_meta.get("ensemble_path") or ""),
                "base_model_paths": [str(p) for p in (ensemble_meta.get("base_model_paths") or [])],
                "loaded_from_cache": bool(ensemble_meta.get("loaded_from_cache", False)),
            }
            if args.metadata == "yes-hash":
                models_section["ensemble_sha256"] = _sha256_ensemble_model(
                    ensemble, models_section["ensemble_path"]
                )

            manifest: Dict[str, object] = {
                "tool": {"name": "spliceai_vector_scan", "version": __version__},
                "versions": {
                    "python": sys.version.split()[0],
                    "numpy": getattr(np, "__version__", None),
                    "tensorflow": tf_ver,
                    "keras": keras_ver,
                },
                "run": {
                    "context": int(args.context),
                    "predict_mode": args.predict_mode,
                    "reverse": bool(do_reverse),
                    "device": {
                        "selected": device_str,
                        "gpu_available": bool(gpu_available),
                        "gpu_disabled": bool(gpu_disabled),
                        "force_cpu": bool(args.cpu),
                    },
                    "determinism": det_level,
                },
                "tf_determinism": det_applied,
                "models": models_section,
                "input": {
                    "dedupe_exact_sequences_for_inference": True,
                    "n_records_raw": int(len(records_raw)),
                    "n_records_unique": int(n_unique),
                },
            }
            write_manifest(manifest_path, manifest)
        except Exception as e:
            logger.warning(f"Failed to write manifest {manifest_path}: {e}")

    # Prepare BED output (single sorted BED6 with acceptor/donor encoded in name)
    bed_rows: Optional[List[Tuple[str, int, int, str, int, str]]] = None
    bed_out_path: Optional[str] = None
    if write_bed_flag:
        bed_rows = []
        bed_out_path = f"{args.out_prefix}.spliceai_bins.bed" + (".gz" if args.gzip else "")

    # Prepare bigWig writers (single file per track) and per-group data for raw-order emission.
    bw_writers = None
    bw_group_data: Optional[List[Optional[Tuple[np.ndarray, Optional[np.ndarray]]]]] = None
    if write_bw_flag:
        chrom_sizes = [(raw_bed_chroms[i], int(len(records_raw[i][1]))) for i in range(len(records_raw))]
        bw_writers = _BigWigTrackWriters(args.out_prefix, chrom_sizes)
        bw_group_data = [None] * n_unique

    # Inference + streaming emission.
    need_members = bool(write_csv_members_flag)
    need_gmean = bool(write_csv_flag)

    logger.info(
        f"Beginning full-length inference over {n_unique} unique sequence(s); "
        f"device={device_str}, mode={args.predict_mode}"
    )

    with tf.device(device_str):
        for g_idx, g in enumerate(groups):
            seq = g["seq"]
            headers = g["headers"]
            merged_header = "|".join(headers)
            merged_id = merged_ids_for_path[g_idx]
            display_label = headers[0] if headers else merged_id
            if len(headers) > 1:
                display_label = f"{display_label} (+{len(headers) - 1} dup)"
            logger.info(
                f"Processing sequence {g_idx + 1}/{n_unique}: {display_label} (len={len(seq):,} bp)"
            )

            m_s, avg_s, g_s = _run_ensemble_one_strand(
                seq=seq,
                header=merged_header,
                strand="sense",
                ensemble_model=ensemble,
                context=args.context,
                predict_mode=args.predict_mode,
                need_members=need_members,
                need_gmean=need_gmean,
            )

            m_rev = avg_rev = g_rev = None
            if do_reverse:
                m_raw, avg_raw, g_raw = _run_ensemble_one_strand(
                    seq=rev(seq),
                    header=merged_header,
                    strand="reverse",
                    ensemble_model=ensemble,
                    context=args.context,
                    predict_mode=args.predict_mode,
                    need_members=need_members,
                    need_gmean=need_gmean,
                )
                avg_rev = avg_raw[::-1, :]
                if g_raw is not None:
                    g_rev = g_raw[::-1, :]
                if m_raw is not None:
                    m_rev = m_raw[::-1, :, :]

            if write_csv_flag:
                path = f"{args.out_prefix}.{merged_id}.sense.csv" + (".gz" if args.gzip else "")
                write_csv(
                    path,
                    record_id=merged_header,
                    strand="sense",
                    probs_mean=avg_s,
                    probs_gmean=g_s,
                    bases=seq,
                    gzip_out=args.gzip,
                )

                if do_reverse and avg_rev is not None:
                    path = f"{args.out_prefix}.{merged_id}.reverse.csv" + (".gz" if args.gzip else "")
                    write_csv(
                        path,
                        record_id=merged_header,
                        strand="reverse",
                        probs_mean=avg_rev,
                        probs_gmean=g_rev,
                        bases=seq,
                        gzip_out=args.gzip,
                    )

            if write_csv_members_flag:
                members_path = f"{args.out_prefix}.{merged_id}.sense.members.csv" + (".gz" if args.gzip else "")
                write_members_csv(
                    members_path,
                    record_id=merged_header,
                    strand="sense",
                    members=m_s,
                    bases=seq,
                    gzip_out=args.gzip,
                )

                if do_reverse and m_rev is not None:
                    members_path = f"{args.out_prefix}.{merged_id}.reverse.members.csv" + (".gz" if args.gzip else "")
                    write_members_csv(
                        members_path,
                        record_id=merged_header,
                        strand="reverse",
                        members=m_rev,
                        bases=seq,
                        gzip_out=args.gzip,
                    )

            # Store per-group outputs for bigWig; emit later in raw input order.
            if write_bw_flag and bw_group_data is not None:
                bw_group_data[g_idx] = (avg_s, avg_rev)

            # bedGraph / BED per *raw* record (duplicate sequences get duplicated tracks/features).
            raw_indices: List[int] = g["raw_indices"]
            for raw_idx in raw_indices:
                chrom_bed = raw_bed_chroms[raw_idx]
                file_id = raw_ids_for_path[raw_idx]

                if write_bg_flag:
                    for cls_idx, cls_name in enumerate(["neither", "acceptor", "donor"]):
                        path = f"{args.out_prefix}.{file_id}.sense.{cls_name}.bedgraph" + (".gz" if args.gzip else "")
                        write_bedgraph(path, chrom=chrom_bed, probs=avg_s, class_index=cls_idx, gzip_out=args.gzip)

                    if do_reverse and avg_rev is not None:
                        for cls_idx, cls_name in enumerate(["neither", "acceptor", "donor"]):
                            path = f"{args.out_prefix}.{file_id}.reverse.{cls_name}.bedgraph" + (".gz" if args.gzip else "")
                            write_bedgraph(path, chrom=chrom_bed, probs=avg_rev, class_index=cls_idx, gzip_out=args.gzip)

                if write_bed_flag and bed_rows is not None:
                    _collect_bed6_bins(bed_rows, chrom_bed, avg_s, class_index=1, strand="+", feature_prefix="ACC:")
                    _collect_bed6_bins(bed_rows, chrom_bed, avg_s, class_index=2, strand="+", feature_prefix="DON:")
                    if do_reverse and avg_rev is not None:
                        _collect_bed6_bins(bed_rows, chrom_bed, avg_rev, class_index=1, strand="-", feature_prefix="ACC:")
                        _collect_bed6_bins(bed_rows, chrom_bed, avg_rev, class_index=2, strand="-", feature_prefix="DON:")

    logger.info("Finished inference for all sequences.")

    # Write sorted BED (track header + sorted intervals)
    if write_bed_flag and bed_rows is not None and bed_out_path is not None:
        opener = gzip.open if args.gzip else open
        bed_rows.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[5]))
        logger.info(f"Writing BED: {bed_out_path}")
        with opener(bed_out_path, "wt", encoding="utf-8", newline="\n") as fh:
            fh.write(
                'track name="SpliceAI_bins" '
                'description="SpliceAI acceptor/donor bins (>=0.05; excludes not_a_site, unlikely_site)" '
                'visibility=2 useScore=1\n'
            )
            for chrom, start, end, name, score, strand in bed_rows:
                fh.write(f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\n")
    elif write_bed_flag and bed_out_path is not None:
        opener = gzip.open if args.gzip else open
        logger.info(f"Writing BED (no intervals passed filter): {bed_out_path}")
        with opener(bed_out_path, "wt", encoding="utf-8", newline="\n") as fh:
            fh.write(
                'track name="SpliceAI_bins" '
                'description="SpliceAI acceptor/donor bins (>=0.05; excludes not_a_site, unlikely_site)" '
                'visibility=2 useScore=1\n'
            )

    # bigWig emission in raw input order
    if write_bw_flag and bw_writers is not None and bw_group_data is not None:
        logger.info("Writing bigWig tracks in raw record order")
        for raw_idx, g_idx in enumerate(raw_to_group):
            rec = bw_group_data[g_idx]
            if rec is None:
                continue
            avg_s, avg_rev = rec
            chrom_bed = raw_bed_chroms[raw_idx]
            bw_writers.add_dense_track(bw_writers.get("sense", "acceptor"), chrom_bed, avg_s[:, 1])
            bw_writers.add_dense_track(bw_writers.get("sense", "donor"),    chrom_bed, avg_s[:, 2])
            if do_reverse and avg_rev is not None:
                bw_writers.add_dense_track(bw_writers.get("reverse", "acceptor"), chrom_bed, avg_rev[:, 1])
                bw_writers.add_dense_track(bw_writers.get("reverse", "donor"),    chrom_bed, avg_rev[:, 2])
        bw_writers.close()

    logger.info("All done!")


if __name__ == "__main__":
    run()
