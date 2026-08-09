from pathlib import Path
import base64


class VisualAssetManager:
    def __init__(
        self,
        output_dir: Path = Path(
            "data/processed/images"
        ),
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_base64_image(
        self,
        image_base64: str,
        filename: str,
    ) -> Path:

        output_path = (
            self.output_dir / filename
        )

        image_bytes = base64.b64decode(
            image_base64
        )

        output_path.write_bytes(
            image_bytes
        )

        return output_path