from setuptools import setup

setup(
    name="tali",
    version="2.8",
    packages=[
        "tali.datasets",
        "tali.models",
        "tali.runner",
        "tali.trainer",
        "tali.utils",
        "base.callbacks",
        "base.utils",
        "base.vendor",
        "base",
        "tali",
    ],
    url="",
    license="GNU General Public License v3.0",
    author="Antreas Antoniou",
    author_email="a.antoniou@ed.ac.uk",
    description="TALI - A multi modal dataset consisting of Temporally correlated Audio, Images (including Video) and Language",
)
