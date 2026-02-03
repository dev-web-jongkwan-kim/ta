from __future__ import annotations

from services.labeling.pipeline import LabelingConfig, run_labeling


def main() -> None:
    config = LabelingConfig()
    spec_hash = run_labeling(config)
    print(f"Labeled universe with spec {spec_hash}")


if __name__ == "__main__":
    main()
