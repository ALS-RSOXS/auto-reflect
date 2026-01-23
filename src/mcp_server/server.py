"""Main MCP server for beamline control."""

from mcp.server.fastmcp import FastMCP

from .tools import ai_tools, dio_tools, motor_tools

mcp = FastMCP("beamline-control")

mcp.tool()(ai_tools.list_ai_channels)
mcp.tool()(ai_tools.get_ai_values)
mcp.tool()(ai_tools.get_ai_with_uncertainty)

mcp.tool()(motor_tools.list_motors)
mcp.tool()(motor_tools.get_motor_positions)
mcp.tool()(motor_tools.get_motor_status)

mcp.tool()(dio_tools.list_dio_channels)
mcp.tool()(dio_tools.get_dio_states)
