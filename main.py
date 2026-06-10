import argparse
# argparse is used to parse CLAs for configuring the server (e.g. port, host, etc.)

from web_app import run_server


def main():
    parser = argparse.ArgumentParser(
        description="Launch the Napari ResView web application using Trame."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to bind the Trame server (0 = auto).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address to bind the server.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    args = parser.parse_args()

    run_server(port=args.port, host=args.host, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
