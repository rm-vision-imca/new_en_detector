from setuptools import setup

package_name = 'energy_tracker'
submodules = "energy_tracker/utils"
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dfg',
    maintainer_email='1074785246@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "energy_tracker_node=energy_tracker.energy_tracker_node:main"
        ],
    },
)
