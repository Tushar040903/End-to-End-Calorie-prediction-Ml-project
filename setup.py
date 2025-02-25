from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''This will return a list of requirements'''
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
    requirements = [req.strip() for req in requirements]
    if "-e ." in requirements:
        requirements.remove("-e .")
    return requirements

setup(
    name='End to End Calorie Prediction ML Project',
    version='0.0.1',
    author='Tushar',
    author_email='tusharbhardwaj9873010398@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
