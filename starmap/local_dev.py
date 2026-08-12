# local_dev.py
#
# LOCAL DEVELOPMENT ONLY -- do not deploy this file.
#
# In production, Nginx serves the static frontend and reverse-proxies /scan,
# /live_feed and /api/ to Gunicorn. There is no Nginx on a laptop, so this
# launcher gives Flask the static-serving job as well, putting the frontend
# and the API on a single origin. That is the only thing the frontend needs:
# main.js uses relative URLs ('/scan', '/data/', and window.location.host for
# the WebSocket), so same-origin means everything resolves with no CORS
# problems and no changes to any deployed file.
#
# Place this file in the same directory as api_server.py (starmap/) and run:
#     cd path\to\znou\starmap
#     python local_dev.py
#
# Then open http://127.0.0.1:5000/discover/
#
# NOTE: api_server.py is imported first and deliberately so. It calls
# gevent.monkey.patch_all() at the top, which must run before other modules
# are imported. Importing it also triggers experiment_runner, which loads
# GPT-2 onto the GPU -- expect a slow first start.

from api_server import app  # noqa: E402  (must be first -- see note above)

import os  # noqa: E402
from flask import send_from_directory, redirect  # noqa: E402

# starmap/ sits inside the site root, so the root is one level up.
SITE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.route("/")
def _local_index():
    return send_from_directory(SITE_ROOT, "index.html")


@app.route("/<path:filename>")
def _local_static(filename):
    """Serve any file under the site root.

    Flask matches static rules before dynamic ones, so the real API routes
    (/scan, /live_feed, /api/exchange/claim, /api/leaderboard) still win.
    This only catches what Nginx would otherwise have served as a file.
    """
    full_path = os.path.join(SITE_ROOT, filename)

    # Directory requests such as /discover/ or /exchange/ get their index.html,
    # which is what Nginx's try_files did in production.
    if os.path.isdir(full_path):
        if not filename.endswith("/"):
            return redirect("/" + filename + "/")
        return send_from_directory(SITE_ROOT, filename + "index.html")

    return send_from_directory(SITE_ROOT, filename)


if __name__ == "__main__":
    print("-" * 62)
    print("  Z-NOU local dev server")
    print(f"  Serving static files from: {SITE_ROOT}")
    print("  Frontend:  http://127.0.0.1:5000/discover/")
    print("  Exchange:  http://127.0.0.1:5000/exchange/")
    print("  Ctrl+C to stop.")
    print("-" * 62)
    # 127.0.0.1 rather than 0.0.0.0: this binds to your machine only, instead
    # of exposing the server to anything else on your network.
    app.run(host="127.0.0.1", port=5000, debug=False)