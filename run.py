from __future__ import annotations

import uvicorn

from backend.app.main import app


if __name__ == "__main__":
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, reload=False)
    server = uvicorn.Server(config)

    # Give the application a safe way to ask Uvicorn to finish the current
    # response and then run its normal lifespan shutdown sequence.
    app.state.request_shutdown = lambda: setattr(server, "should_exit", True)
    server.run()
