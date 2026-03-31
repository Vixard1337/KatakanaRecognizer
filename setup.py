from setuptools import setup, find_packages

setup(
    name='katakana_recognition_project',
    version='0.1.0',
    description='A program for recognizing Japanese Katakana characters using neural networks',
    author='Konrad Klimek',
    author_email='your.email@example.com',
    url='https://github.com/vamper1337/katakana_recognition',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'tensorflow',  # or 'torch' if you are using PyTorch
        'scikit-learn',
        'matplotlib',
        'opencv-python'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)