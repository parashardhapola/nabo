from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


if __name__ == "__main__":

    CLASSIFIERS = [
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Visualization',
    ]
    KEYWORDS = ['scRNA-Seq', 'bioinformatics', 'single cell', 'analysis',
                'graph', 'cell alignment', 'rna-seq', 'sequencing',
                'shared nearest neighbors', 'knn', 'snn']
    VERSION = open('VERSION').readline().rstrip('\n')
    setup(
        name='nabo',
        description='Python library to perform memory efficient cross-sample cell mapping using single cell transciptomics (scRNA-Seq) data',
        long_description=read('README.rst'),
        author='Parashar Dhapola',
        author_email='parashar.dhapola@gmail.com',
        maintainer='Parashar Dhapola',
        maintainer_email='parashar.dhapola@gmail.com',
        url='https://github.com/parashardhapola/nabo',
        license='BSD 3-Clause',
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        install_requires=[x.strip() for x in open('requirements.txt')],
        version=VERSION,
        packages=find_packages(),
        include_package_data=True,
        scripts=[
            'bin/nabo',
        ],
    )
