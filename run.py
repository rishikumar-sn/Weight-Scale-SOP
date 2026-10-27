from __future__ import annotations

import logging
import threading
import webbrowser

import uvicorn

from backend.app.main import app


class BrowserServer(uvicorn.Server):
    """Open the local UI once, after model loading and socket startup finish."""

    async def startup(self, sockets=None):
        await super().startup(sockets=sockets)
        if self.started and not self.should_exit:
            threading.Thread(target=self.open_ui, daemon=True).start()

    def open_ui(self):
        # Existing tabs reconnect their live socket/poll after a backend restart.
        # Allow those connections to arrive before deciding to open another tab.
        if app.state.browser_ui_seen.wait(timeout=5):
            logging.getLogger("uvicorn.error").info("UI is already open; keeping the existing browser tab.")
            return
        if self.should_exit:
            return
        url = "http://localhost:8000"
        try:
            if webbrowser.open(url, new=2):
                return
        except Exception:
            logging.getLogger("uvicorn.error").warning("Could not open the default browser.", exc_info=True)
        logging.getLogger("uvicorn.error").warning("Open %s in your browser to use the app.", url)


if __name__ == "__main__":
    app.state.browser_ui_seen = threading.Event()
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, reload=False)
    server = BrowserServer(config)

    # Give the application a safe way to ask Uvicorn to finish the current
    # response and then run its normal lifespan shutdown sequence.
    app.state.request_shutdown = lambda: setattr(server, "should_exit", True)
    server.run()
