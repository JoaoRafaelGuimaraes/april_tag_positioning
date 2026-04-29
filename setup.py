from setuptools import find_packages, setup

package_name = 'april_tag_positioning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
        'matplotlib',
        'dash',
        'plotly',
    ],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='joaorafaelguimaraes@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'video_camera_pub = april_tag_positioning.video_camera_pub:main',
            'tag_listener = april_tag_positioning.tag_listener:main',
            'tag_listener_v2 = april_tag_positioning.tag_listener_v2:main',
            'tf_publisher = april_tag_positioning.tf_publisher:main',
        ],
    },
)
