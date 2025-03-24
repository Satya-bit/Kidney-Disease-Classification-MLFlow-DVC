#This will cerate local packages like from src.cnnClassifier.components like this.
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read() #This reads the contents of README.md, which will be used as the long description of the package.
    #For example suppose we post this on PyPi then it will be the long description of the package taken from GitHub README




__version__ = "0.0.0"

REPO_NAME = "Kidney-Disease-Classification-MLFlow-DVC"
AUTHOR_USER_NAME = "Satya-bit"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "satya.s@ahduni.edu.in"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)