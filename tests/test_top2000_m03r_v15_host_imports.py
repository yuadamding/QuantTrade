from __future__ import annotations

import subprocess
import sys


def test_v15_host_lifecycle_imports_no_training_data_stack() -> None:
    command = (
        "import sys; "
        "import rl_quant.training.top2000_m03r_v15_seadragon_lifecycle; "
        "assert 'torch' not in sys.modules; "
        "assert 'pyarrow' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
