from setuptools import setup, find_packages

setup(
    name="MLlib",
    version='0.0.1',
    description='pytorch library to build ML models',
    author='duytinvo',
    author_email='duytinvo@gmail.com',
    packages=find_packages(),
    include_package_data=True
)

# python setup.py sdist
# twine upload -r testpypi dist/MLlib-0.0.1.tar.gz
# twine upload dist/MLlib-0.0.1.tar.gz
