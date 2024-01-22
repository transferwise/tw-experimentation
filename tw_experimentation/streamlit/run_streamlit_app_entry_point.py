import os
import runpy
import sys

import tw_experimentation.streamlit


def main() -> None:
    this_file = os.path.dirname(tw_experimentation.streamlit.__file__)
    streamlit_script_path = os.path.join(this_file, "Main.py")
    sys.argv = ["streamlit", "run", streamlit_script_path]
    runpy.run_module("streamlit", run_name="__main__")


if __name__ == "__main__":
    main()
