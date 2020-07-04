from io import open
import setuptools

long_description = "NLP toolkit containing state-of-the-art models for various NLP tasks"
with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="nlptoolkit",
    version="0.0.18",
    author="Soh Wee Tee",
    author_email="weeteesoh345@gmail.com",
    description="NLP toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="NLP pytorch AI deep learning",
    licence="Apache",
    url="https://github.com/plkmo/NLP_Toolkit",
    packages=setuptools.find_packages(exclude=["data"\
                                               "results",\
                                               ]),
    install_requires=required,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
