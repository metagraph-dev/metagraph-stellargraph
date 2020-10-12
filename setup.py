from setuptools import setup, find_packages
import versioneer

setup(
    name="metagraph-stellargraph",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Stellargraph plugins for Metagraph",
    author="Anaconda, Inc.",
    packages=find_packages(
        include=["metagraph_stellargraph", "metagraph_stellargraph.*"]
    ),
    include_package_data=True,
    install_requires=["metagraph", "stellargraph"],
    entry_points={
        "metagraph.plugins": "plugins=metagraph_stellargraph.plugins:find_plugins"
    },
)
