from .geo import RasterDataset

class RasterizedEuroCrops(RasterDataset):
    filename_glob = "*.tif"
    filename_regex = r"""
        ^(?P<date>\d{4})
        _.*$
    """
    date_format = "%Y"
    is_image = False
