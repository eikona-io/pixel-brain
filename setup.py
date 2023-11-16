from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='pixel_brain',
    version='0.1',
    packages=find_packages(exclude=["tests", "tests/*"]),
    description='A package for procssing image data using ML models',
    author='Omer Hacohen',
    author_email='omerhac94@gmail.com',
    url='https://github.com/omerhac/pixel-brain.git',  # URL of your package's source code
    install_requires=required,
    entry_points={
        'apps': [
            'tag_identity = pixel_brain.apps.tag_identity:main',
        ],
    },
    python_requires='>=3.10',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)