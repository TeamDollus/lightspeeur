from setuptools import find_packages
from setuptools import setup

setup(name='lightspeeur',
      version='0.0.10',
      description='Lightspeeur TensorFlow Mode Development Framework',
      url='https://github.com/TeamDollus/lightspeeur.git',
      author='OrigamiDream',
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True,
      install_requires=[
            'tensorflow>=2.0',
            'numpy',
            'tqdm',
      ])
