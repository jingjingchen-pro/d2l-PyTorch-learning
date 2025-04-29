from setuptools import setup, find_packages
import d2l

requirements = [
    'jupyter==1.0.0',
    'numpy==1.21.5',
    'matplotlib==3.5.1',
    'requests==2.25.1',
    'pandas==1.3.5'  # [250428]1.2.4 -> 1.3.5 because I use python3.10 in mac, in windows, 1.2.4 works fine
]

setup(
    name='d2l',
    version=d2l.__version__,
    python_requires='>=3.5',
    author='D2L Developers',
    author_email='d2l.devs@gmail.com',
    url='https://d2l.ai',
    description='Dive into Deep Learning',
    license='MIT-0',
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
)
