import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


long_description = read('README.md') if os.path.isfile("README.md") else ""

setup(
    name='ethereum-etl',
    version='2.4.2',
    author='Evgeny Medvedev',
    author_email='evge.medvedev@gmail.com',
    description='Tools for exporting Ethereum blockchain data to CSV or JSON',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/blockchain-etl/ethereum-etl',
    packages=find_packages(exclude=['schemas', 'tests']),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    keywords='ethereum',
    python_requires='>=3.7.2,<4',
    install_requires=[
        'web3>=6,<7',
        'eth-utils>=4',
        'eth-abi>=5',
        'python-dateutil>=2.8.0,<3',
        'click>=8.0.4,<9',
        'ethereum-dasm==0.1.4',
        'urllib3',
        'base58',
        'requests',
        'setuptools<81',
    ],
    extras_require={
        'streaming': [
            'timeout-decorator>=0.4.1',
            'google-cloud-pubsub>=2.13.0',
            'google-cloud-storage>=1.33.0',
            'kafka-python>=2.0.2',
            'sqlalchemy>=1.4,<2',
            'pg8000>=1.16.6',
            'grpcio>=1.46.3',
        ],
        'streaming-kinesis': [
            'boto3',
            'botocore',
        ],
        'dev': [
            'pytest~=4.3.0'
        ]
    },
    entry_points={
        'console_scripts': [
            'ethereumetl=ethereumetl.cli:cli',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/blockchain-etl/ethereum-etl/issues',
        'Chat': 'https://gitter.im/ethereum-etl/Lobby',
        'Source': 'https://github.com/blockchain-etl/ethereum-etl',
    },
)
