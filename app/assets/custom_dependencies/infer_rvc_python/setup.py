import os
import platform
from setuptools import find_packages, setup

setup(
    name="infer_rvc_python",
    version="1.2.0",
    description="Python wrapper for fast inference with rvc",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    readme="README.md",
    python_requires=">=3.10",
    author="R3gm",
    url="https://github.com/R3gm/infer_rvc_python",
    license="MIT",
    packages=find_packages(),
    package_data={'': ['*.txt', '*.rep', '*.pickle']},
    install_requires=[
        "torch",
        "torchaudio",
        "gradio",
        "praat-parselmouth>=0.4.3",
        "pyworld==0.3.2",
        "faiss-cpu==1.7.3",
        "torchcrepe>=0.0.20",  # Allow newer versions
        "ffmpeg-python>=0.2.0",
        "fairseq==0.12.2",
        "typeguard==4.2.0",
        "soundfile",
        "librosa>=0.10.1",  # Updated to support NeMo compatibility
        "numpy<2.0",  # faiss-cpu 1.7.3 requires NumPy <2.0
    ],
    include_package_data=True,
    extras_require={"all": [
        "scipy",
        "numba==0.56.4",
        "edge-tts"
        ]},
)