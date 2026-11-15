import logging
import sys


def create_repro_logger(train_seed, env_seed, log_dir):
    logger = logging.getLogger(f"repro_{train_seed}_{env_seed}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # prevents global leak

    # Remove any existing handlers (important for repeated runs)
    logger.handlers.clear()

    # File handler (DEBUG only)
    fh = logging.FileHandler(f"{log_dir}/repro_seed_{train_seed}.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(f"TRAIN={train_seed} ENV={env_seed} | %(message)s"))
    logger.addHandler(fh)

    # Stdout handler (INFO only)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(f"TRAIN={train_seed} ENV={env_seed} | %(message)s"))
    logger.addHandler(sh)

    return logger


def create_run_logger(run_dir, console):
    """Local rotating diagnostics; metrics are written independently as JSONL."""
    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    from rich.logging import RichHandler

    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"dreamerrl.run.{path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    file_handler = RotatingFileHandler(path / "run.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(RichHandler(console=console, show_path=False, markup=False))
    return logger
