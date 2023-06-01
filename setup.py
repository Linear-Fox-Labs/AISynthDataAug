from setuptools import setup, find_packages

setup(
    name='ai-image-synthesis',
    version='0.0.0',
    description='AI-driven image synthesis tool for data augmentation',
    author='Nathan Fargo',
    author_email='ntfargo@linearfox.com',
    url='https://github.com/Linear-Fox-Labs/AISynthDataAug',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'tensorflow',
        'pillow',
        'scikit-image',
        'opencv-python',
        'tqdm'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)