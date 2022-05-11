from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='sklearnEASE',
    version='0.0.0',
    author='Ola Oni',
    author_email='oa.oni7@gmail.com',
    description='Sklearn wrapper for EASE - Embarrassingly Shallow Autoencoders (Harald Steck)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/oaoni/sklearnEASE',
    project_urls = {
        "Bug Tracker": "https://github.com/oaoni/sklearnEASE/issues"
    },
    license='MIT',
    packages=find_packages(include=['sklearnEASE','sklearnEASE.*']),
    install_requires=['scipy','sklearn','pandas','numpy'],
)
