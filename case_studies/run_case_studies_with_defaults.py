import logging
import warnings
from pathlib import Path

from case_studies.comparison_utils import compare_with_expected, compare_with_expected_by_size
from case_studies.duffing_oscillator.main_duffing_oscillator_with_defaults import (
    main as duffing_main,
)
from case_studies.friction.main_friction_with_defaults import main as friction_main
from case_studies.lorenz.main_lorenz_with_defaults import main as lorenz_main
from case_studies.pendulum.main_pendulum_case1_with_defaults import main as pendulum_main

warnings.filterwarnings("ignore", message="No cluster_classifier provided")
warnings.filterwarnings("ignore", message="os.fork\\(\\) was called")


logging.getLogger("pybasin").setLevel(logging.WARNING)

TESTS_DIR = Path(__file__).parent.parent / "tests" / "integration"


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DUFFING OSCILLATOR")
    print("=" * 60)
    bse = duffing_main()
    if bse.bs_vals is not None:
        compare_with_expected_by_size(
            bse.bs_vals, TESTS_DIR / "duffing" / "main_duffing_supervised.json"
        )

    print("\n" + "=" * 60)
    print("FRICTION")
    print("=" * 60)
    bse = friction_main()
    if bse.bs_vals is not None:
        compare_with_expected(
            bse.bs_vals, {"0": "LC", "1": "FP"}, TESTS_DIR / "friction" / "main_friction_case1.json"
        )

    print("\n" + "=" * 60)
    print("PENDULUM")
    print("=" * 60)
    bse = pendulum_main()
    if bse.bs_vals is not None:
        compare_with_expected(
            bse.bs_vals, {"0": "LC", "1": "FP"}, TESTS_DIR / "pendulum" / "main_pendulum_case1.json"
        )

    print("\n" + "=" * 60)
    print("LORENZ")
    print("=" * 60)
    bse = lorenz_main()
    if bse.bs_vals is not None:
        compare_with_expected(
            bse.bs_vals,
            {"0": "butterfly1", "1": "butterfly2", "unbounded": "unbounded"},
            TESTS_DIR / "lorenz" / "main_lorenz.json",
        )

    print("\n" + "=" * 60)
    print("ALL CASE STUDIES COMPLETED")
    print("=" * 60)
