from pathlib import Path


def test_surface_parity_doc_explains_the_sdk_boundary():
    text = Path("docs/advanced/surface-parity.md").read_text()

    assert 'build_product_tools(surface="gateway", ...)' in text
    assert 'build_product_tools(surface="web", ...)' not in text
    assert "## Python SDK" in text
    assert "from agentica import Agent" in text
    assert "WorktreeTool(WorktreeBinder(agent_config=" in text