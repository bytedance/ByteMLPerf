# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


from setuptools import setup, find_packages
from dltest.cli.entry_points import make_execute_path

setup(
    name="dltest",
    version="0.1",
    description='Iluvatar Corex AI Toolbox',
    packages=find_packages(exclude=('examples')),
    include_package_data=True,
    zip_safe=False,
    entry_points = {
        'console_scripts': make_execute_path(),
    },
    install_requires=[
        'psutil'
    ]
)
