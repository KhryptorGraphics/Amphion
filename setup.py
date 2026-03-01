# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

setup(
    name="amphion",
    version="0.1.0",
    description="Amphion: Open-Source Audio, Music, and Speech Generation toolkit.",
    packages=find_packages(exclude=["tests*", "egs*", "data*", "ckpts*"]),
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "amphion=cli.main:main",
        ],
    },
)
