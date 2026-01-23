"""Entry point for running MCP server as standalone application."""

import sys

from .server import mcp

if __name__ == "__main__":
    try:
        mcp.run()
    except KeyboardInterrupt:
        sys.exit(0)
