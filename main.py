import json
import pickle
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import torch
from PIL import Image
from fire import Fire
from pydantic import BaseModel
from torchvision.datasets.utils import download_and_extract_archive

from fromage.models import load_fromage, Fromage
from fromage.utils import get_image_from_url


class MultimodalPart(BaseModel):
    is_image: bool = False
    is_audio: bool = False
    content: str

    @property
    def is_path(self) -> bool:
        return Path(self.content).exists()

    @property
    def is_url(self) -> bool:
        return urlparse(self.content).scheme != ""


class MultimodalSequence(BaseModel):
    parts: List[MultimodalPart]


class MultimodalModel(BaseModel):
    def run(self, context: MultimodalSequence) -> MultimodalSequence:
        raise NotImplementedError


class FromageModel(MultimodalModel, arbitrary_types_allowed=True):
    model_dir: str = "fromage_model"
    model: Optional[Fromage]

    @staticmethod
    def download_checkpoint(
        url: str = "https://github.com/chiayewken/multimodal-inference/releases/download/v0.1.0/fromage_model.zip",
        path: str = ".",
        folder: str = "fromage_model",
        embed_path: str = "cc3m_embeddings.pkl",
    ):
        download_and_extract_archive(url, download_root=path)
        assert Path(folder).exists()

        # Write dummy image embeddings needed to load model
        data = dict(paths=[""], embeddings=torch.zeros(1, 256))
        with open(Path(folder, embed_path), "wb") as f:
            pickle.dump(data, f)

    def load(self):
        if self.model is None:
            self.download_checkpoint(
                path=str(Path(self.model_dir).parent.resolve()),
                folder=self.model_dir,
            )
            self.model = load_fromage(self.model_dir)

    def run(self, context: MultimodalSequence) -> MultimodalSequence:
        inputs = []
        for part in context.parts:
            if part.is_image:
                if part.is_url:
                    inputs.append(get_image_from_url(part.content))
                elif part.is_path:
                    image = Image.open(part.content)
                    image = image.resize((224, 224))
                    image = image.convert("RGB")
                    inputs.append(image)
                else:
                    raise ValueError(str(part))
            else:
                inputs.append(part.content)

        self.load()
        outputs = self.model.generate_for_images_and_texts(
            inputs, num_words=32, ret_scale_factor=0.0  # Don't generate images
        )
        return MultimodalSequence(parts=[MultimodalPart(content=outputs[0])])


def test_fromage(
    image_paths: List[str] = (
        "https://i.pinimg.com/736x/d3/8c/21/d38c21ca670ce0be2d01c301b1f0e7d3--vintage-dior-vintage-dresses.jpg",
        "https://secure.img1-cg.wfcdn.com/im/06305386/compr-r85/1695/169530576/ainsley-736-vegan-leather-sofa.jpg",
    ),
    prompts: List[str] = (
        "Q: What is this image?\nA:",
        "Q: What color is the dress?\nA:",
        "Q: When do you think it was taken?\nA:",
        "Q: What color is the sofa?\nA:",
        "Q: What is the difference between the two images?\nA:",
    ),
):
    print(json.dumps(locals(), indent=2))
    parts = [MultimodalPart(content=path, is_image=True) for path in image_paths]
    model = FromageModel()

    for p in prompts:
        outputs = model.run(
            context=MultimodalSequence(parts=parts + [MultimodalPart(content=p)]),
        )
        print(dict(prompt=p, outputs=outputs.parts[0].content))

    """
    """


if __name__ == "__main__":
    Fire()
