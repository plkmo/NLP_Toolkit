import setuptools

long_description = "NLP toolkit containing state-of-the-art models for various NLP tasks"
setuptools.setup(
    name="nlptoolkit-plkmo",
    version="0.0.1",
    author="Soh Wee Tee",
    author_email="weeteesoh345@gmail.com",
    description="NLP toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/plkmo/NLP_Toolkit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
