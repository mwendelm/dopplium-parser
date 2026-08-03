"""Shared verbosity gating for the binary parse functions.

Verbosity is an int in [0, 4]: 0 prints nothing, 4 prints everything.
Anomaly warnings and EOF/struct-error notices are not gated by this scale
and always print, independent of verbosity.
"""


def _validate_verbose(verbose: int) -> int:
    verbose = int(verbose)
    if not 0 <= verbose <= 4:
        raise ValueError(f"verbose must be an int in [0, 4], got {verbose!r}")
    return verbose
