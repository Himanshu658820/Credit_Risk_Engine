"""Repository-root launcher for Streamlit deployments.

Hugging Face Spaces expects an app entrypoint at the repository root,
so this wrapper delegates to the existing dashboard implementation.
"""

from dashboard.app import *  # noqa: F401,F403