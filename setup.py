import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
with open('requirements.txt', 'r') as req:
    packages = req.read().splitlines()
    try:
        packages.remove('')
    except:
        pass

setuptools.setup(
    name='BERN2',
    version='0.1.1',
    scripts=[],
    author="Navina ai",
    author_email="tech@navina.ai",
    description="BERN2",
    long_description="BERN2",
    long_description_content_type="text/markdown",
    install_requires=packages,
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
