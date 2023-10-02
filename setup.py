from setuptools import find_packages, setup

def read_requirements():
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    return requirements
    
if __name__ == '__main__':
    setup(
        name='exlib',
        packages=['exlib'],
        install_requires=read_requirements(),
    )
