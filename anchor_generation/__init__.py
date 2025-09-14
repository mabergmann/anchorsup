"""Anchor generation service package."""
from typing import Dict, Any

from upright_anchor.anchor_generation.base import IAnchorGenerationService
from upright_anchor.anchor_generation.icosahedron_tessellation import IcosahedronTessellation
from upright_anchor.anchor_generation.single_anchor import SingleAnchor


def create_anchor_generator(config: Dict[str, Any]) -> IAnchorGenerationService:
    """Create an anchor generator based on the configuration.

    Args:
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        IAnchorGenerationService: The anchor generator.

    Raises:
        ValueError: If the anchor generator type is not supported.
    """
    generator_type = config["model"].get("anchor_generator", "icosahedron")
    n_anchors = config["model"]["n_anchors"]

    if generator_type == "icosahedron":
        return IcosahedronTessellation(n_anchors)
    elif generator_type == "single_anchor":
        return SingleAnchor(n_anchors)
    else:
        raise ValueError(f"Unsupported anchor generator type: {generator_type}")
