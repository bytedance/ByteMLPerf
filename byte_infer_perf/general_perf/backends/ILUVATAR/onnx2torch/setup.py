import setuptools


import setuptools  # noqa

setuptools.setup(
    name="onnx2torch",
    author="ENOT LLC",
    version="1.15.4",
    author_email="enot@enot.ai",
    packages=setuptools.find_packages(where="onnx2torch"),
    python_requires=">=3.6",
    install_requires=[
        'numpy>=1.16.4',
        'onnx>=1.9.0',
        'torch>=1.8.0',
        'torchvision>=0.9.0',
    ],
    entry_points={
        'console_scripts': [
            'apap = package.main:main',
        ]

    }
)

