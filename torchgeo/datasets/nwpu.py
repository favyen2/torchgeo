import os
from typing import Any, Callable, Dict, Optional

from PIL import Image
from torchvision.datasets.utils import (
    check_integrity,
    download_file_from_google_drive,
    download_url,
)

from .geo import VisionDataset


class VHR10(VisionDataset):
    """Northwestern Polytechnical University (NWPU) very-high-resolution ten-class
    (VHR-10) remote sensing image dataset.

    Consists of 800 VHR optical remote sensing images, where 715 color images were
    acquired from Google Earth with the spatial resolution ranging from 0.5 to 2 m,
    and 85 pansharpened color infrared (CIR) images were acquired from Vaihingen data
    with a spatial resolution of 0.08 m.

    The data set is divided into two sets:

    * Positive image set (650 images) which contains at least one target in an image
    * Negative image set (150 images) does not contain any targets

    The positive image set consists of objects from ten classes:

    1. Airplanes (757)
    2. Ships (302)
    3. Storage tanks (655)
    4. Baseball diamonds (390)
    5. Tennis courts (524)
    6. Basketball courts (159)
    7. Ground track fields (163)
    8. Harbors (224)
    9. Bridges (124)
    10. Vehicles (477)

    Includes object detection bounding boxes from original paper and instance
    segmentation masks from follow-up publications. If you use this dataset in your
    research, please cite the following papers:

    * https://doi.org/10.1016/j.isprsjprs.2014.10.002
    * https://doi.org/10.1109/IGARSS.2019.8898573
    * https://doi.org/10.3390/rs12060989

    .. note::

       This dataset requires the following additional libraries to be installed:

       * `pycocotools <https://pypi.org/project/pycocotools/>`_ to load the
         ``annotations.json`` file for the "positive" image set
       * `rarfile <https://pypi.org/project/rarfile/>`_ to extract the dataset,
         which is stored in a RAR file
    """

    base_folder = "vhr10"
    image_meta = {
        "file_id": "1--foZ3dV5OCsqXQXT84UeKtrAqc5CkAE",
        "filename": "NWPU VHR-10 dataset.rar",
        "md5": "d30a7ff99d92123ebb0b3a14d9102081",
    }
    target_meta = {
        "url": (
            "https://raw.githubusercontent.com/chaozhong2010/VHR-10_dataset_coco/"
            "master/NWPU%20VHR-10_dataset_coco/annotations.json"
        ),
        "filename": "annotations.json",
        "md5": "7c76ec50c17a61bb0514050d20f22c08",
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "positive",
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
    ) -> None:
        """Initialize a new VHR-10 dataset instance.

        Parameters:
            root: root directory where dataset can be found
            split: one of "postive" or "negative"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        assert split in ["positive", "negative"]

        self.root = root
        self.split = split
        self.transforms = transforms

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        if split == "positive":
            # Must be installed to parse annotations file
            from pycocotools.coco import COCO

            self.coco = COCO(
                os.path.join(
                    self.root,
                    self.base_folder,
                    "NWPU VHR-10 dataset",
                    self.target_meta["filename"],
                )
            )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.

        Parameters:
            index: index to return

        Returns:
            data and label at that index
        """
        id_ = index % len(self) + 1
        sample = {
            "image": self._load_image(id_),
            "label": self._load_target(id_),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        if self.split == "positive":
            return 650
        else:
            return 150

    def _load_image(self, id_: int) -> Image.Image:
        """Load a single image.

        Parameters:
            id_: unique ID of the image

        Returns:
            the image
        """
        return Image.open(
            os.path.join(
                self.root,
                self.base_folder,
                "NWPU VHR-10 dataset",
                self.split + " image set",
                f"{id_:03d}.jpg",
            )
        ).convert("RGB")

    def _load_target(self, id_: int) -> Dict[str, Any]:
        """Load the annotations for a single image.

        Parameters:
            id_: unique ID of the image

        Returns:
            the annotations
        """
        # Images in the "negative" image set have no annotations
        annot = []
        if self.split == "positive":
            annot = self.coco.loadAnns(self.coco.getAnnIds(id_))

        target = dict(image_id=id_, annotations=annot)

        return target

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset MD5s match, else False
        """
        image: bool = check_integrity(
            os.path.join(self.root, self.base_folder, self.image_meta["filename"]),
            self.image_meta["md5"],
        )

        # Annotations only needed for "positive" image set
        target = True
        if self.split == "positive":
            target = check_integrity(
                os.path.join(
                    self.root,
                    self.base_folder,
                    "NWPU VHR-10 dataset",
                    self.target_meta["filename"],
                ),
                self.target_meta["md5"],
            )

        return image and target

    def download(self) -> None:
        """Download the dataset and extract it."""

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        # Download images
        download_file_from_google_drive(
            self.image_meta["file_id"],
            os.path.join(self.root, self.base_folder),
            self.image_meta["filename"],
            self.image_meta["md5"],
        )

        # Must be installed to extract RAR file
        import rarfile

        with rarfile.RarFile(
            os.path.join(self.root, self.base_folder, self.image_meta["filename"])
        ) as f:
            f.extractall(os.path.join(self.root, self.base_folder))

        # Annotations only needed for "positive" image set
        if self.split == "positive":
            # Download annotations
            download_url(
                self.target_meta["url"],
                os.path.join(self.root, self.base_folder, "NWPU VHR-10 dataset"),
                self.target_meta["filename"],
                self.target_meta["md5"],
            )
