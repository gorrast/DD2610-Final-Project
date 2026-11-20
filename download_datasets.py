import hashlib
from http.client import HTTPException
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4


def _download(destination, url, algorithm, checksum):
    if destination.is_file():
        return

    print(f"Downloading {destination.name}", flush=True)
    partial = destination.with_name(f".{destination.name}.{uuid4().hex}.part")
    digest = hashlib.new(algorithm)
    try:
        with urlopen(url, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)

        if digest.hexdigest() != checksum:
            raise RuntimeError(f"Checksum mismatch for {destination.name}.")
        partial.replace(destination)
    except (OSError, HTTPException, ValueError) as error:
        raise RuntimeError(
            f"Could not download {destination.name} ({type(error).__name__})."
        ) from None
    finally:
        partial.unlink(missing_ok=True)


def download_datasets(drive_folder: Path, imagenet_val_url: str | None = None):
    datasets = drive_folder / "datasets"
    datasets.mkdir(exist_ok=True)

    _download(
        datasets / "256_ObjectCategories.tar",
        "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1",
        "md5",
        "67b4f42ca05d46448c6bb8ecd2220f6d",
    )
    _download(
        datasets / "imagenetv2-matched-frequency.tar.gz",
        "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/"
        "d626240be2538720e83103a0e1178d24aca8b12c/imagenetv2-matched-frequency.tar.gz",
        "sha256",
        "f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c",
    )
    _download(
        datasets / "ILSVRC2012_devkit_t12.tar.gz",
        "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz",
        "md5",
        "fa75699e90414af021442c21a62c3abf",
    )

    validation = datasets / "ILSVRC2012_img_val.tar"
    if imagenet_val_url:
        _download(validation, imagenet_val_url, "md5", "29b22e2961454d5413ddabcf34fc5622")
    elif not validation.is_file():
        print("ImageNet-1K validation is pending. Use IMAGENET_VAL_URL with a team-restricted folder.")

    print(f"Dataset folder: {datasets}")
