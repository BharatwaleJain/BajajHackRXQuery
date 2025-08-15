from setuptools import setup, find_packages
setup(
    name='hackrx_api',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
    ],
    entry_points={
        'console_scripts': [
            'hackrx-api=app.main:app',
        ],
    },
)