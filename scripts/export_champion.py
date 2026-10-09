"""Export the current champion from the MLflow registry to models/champion/.

Usage (from the repository root), before building the Docker image:
    uv run python scripts/export_champion.py
"""

from fraudguard.api.model_service import export_model, load_champion
from fraudguard.config import PROJECT_ROOT, load_settings


def main() -> None:
    service = load_champion(load_settings())
    out_dir = PROJECT_ROOT / "models" / "champion"
    export_model(service, out_dir)
    print(
        f"Exported {service.name} v{service.version} (threshold {service.threshold}) to {out_dir}"
    )


if __name__ == "__main__":
    main()
