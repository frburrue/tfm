import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras-yolo3-pck-qqwweee",
    version="1.0",
    author="qqwweee",
    description="A Keras implementation of YOLOv3 (Tensorflow backend) inspired by allanzelener/YAD2K",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qqwweee/keras-yolo3",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)