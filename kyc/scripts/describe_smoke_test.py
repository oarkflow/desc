import asyncio
import sys
from pathlib import Path

from ultralytics import ASSETS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from kyc.describe.service import describe_image, health
except ModuleNotFoundError:
    from describe.service import describe_image, health


class UploadStub:
    def __init__(self, content: bytes, content_type: str):
        self._content = content
        self.content_type = content_type

    async def read(self):
        return self._content


async def main():
    status = await health()
    print(f"health={status}")

    sample = Path(ASSETS) / "bus.jpg"
    response = await describe_image(UploadStub(sample.read_bytes(), "image/jpeg"))
    top_objects = [item["label"] for item in response["objects"][:5]]

    print(f"sample={sample}")
    print(f"caption={response['caption']}")
    print(f"top_objects={top_objects}")
    print(f"object_count={response['object_count']}")
    print(f"tags={response['tags']}")
    print(f"text={response['text']}")
    print(f"text_languages={response['text_languages']}")
    print(f"tamper={response['tamper']['verdict']} score={response['tamper']['score']}")


if __name__ == "__main__":
    asyncio.run(main())
