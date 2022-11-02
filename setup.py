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
    version='0.2.4',
    scripts=[],
    author="Navina ai",
    author_email="tech@navina.ai",
    description="BERN2",
    long_description="BERN2",
    long_description_content_type="text/markdown",
    install_requires=packages,
    packages=["temp_name_021122", "temp_name_021122/bern2", "temp_name_021122/multi_ner",
              "temp_name_021122/normalizers"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
