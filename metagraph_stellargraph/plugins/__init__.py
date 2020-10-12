############################
# Libraries used as plugins
############################

try:
    import stellargraph as _

    has_stellargraph = True
except ImportError:
    has_stellargraph = False


import metagraph

# Use this as the entry_point object
registry = metagraph.PluginRegistry("metagraph_cuda")


def find_plugins():
    # Ensure we import all items we want registered
    from . import stellargraph

    registry.register_from_modules(cudf, name="metagraph_stellargraph_stellargraph")
    return registry.plugins
