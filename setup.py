from setuptools import Command, find_packages, setup

__lib_name__ = "LETSmix"
__lib_version__ = "1.1.1"
__description__ = "LETSmix: a spatially informed and learning-based domain adaptation method for cell-type deconvolution in spatial"
__url__ = "https://github.com/ZhanYangen/LETSmix"
__author__ = "Yangen Zhan"
__author_email__ = "zygmail378@163.com"
__license__ = "MIT"
__keywords__ = ["Spatial transcriptomics", "Domain adaptation", "Cell type deconvolution", "Single-cell RNA-seq"]
__requires__ = ["requests",]

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ["LETSmix"],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
)
