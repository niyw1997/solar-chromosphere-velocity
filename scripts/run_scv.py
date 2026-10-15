#!/usr/bin/env python3
"""Run the processor from a source checkout without installing it.

Equivalent to the ``scv`` console script; kept so the workflow in this
repository does not depend on the package being pip-installed.
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from solar_chromosphere_velocity.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
