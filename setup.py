import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='freefft',
    version='0.0.8',
    author='Frank Pereny',
    author_email='fjpereny@gmail.com',
    description='Fast Fourier Transform (FFT) analyzer and condition monitoring software.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fjpereny/freeFFT',
    packages=setuptools.find_packages(),
    
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],

    python_requires='>=3.6',
    
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'pyqtgraph',
        'PyQt5',
    ],

    keywords='free fft freefft vibration condition monitoring fast fourier transform signal analysis',

    project_urls={
        'Homepage': 'https://www.frankpereny.com/freefft/',
        'Git Repository': 'https://github.com/fjpereny/freeFFT',
        'Bug Reporting': 'https://github.com/fjpereny/freeFFT/issues',
    },

    entry_points={
        'console_scripts': [
            'freefft=freefft.__main__:main',
        ],
    },
)