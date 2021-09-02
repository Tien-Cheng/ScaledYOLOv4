from setuptools import setup, find_packages

setup(name='scaledyolov4',
      version='1.0',
      packages=find_packages(),
      package_data={
        'configs': ['*.yaml'],
        'data': ['*.yaml'],
        'weights': ['*.pt']
      })
