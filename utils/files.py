"""Server-folder / upload file input, unique names, pickle and data reading,
safe literal parsing.
"""

import ast
import hashlib
import io
import numbers
import os
import pickle
from typing import Literal, cast, overload

import numpy as np
import streamlit as st

# --- FILE INPUT WIDGET (upload OR scan a server folder) ---


class LocalFile(io.BytesIO):
    """A file read from disk that mimics Streamlit's ``UploadedFile``.

    It is a ``BytesIO`` subclass exposing ``.name``, ``.size`` and ``.type``,
    so it is a drop-in for the objects returned by ``st.file_uploader`` and can
    be passed directly to ``pd.read_csv``, ``json.load``, ``pickle.load`` etc.
    """

    def __init__(self, path: str):
        with open(path, "rb") as fh:
            data = fh.read()
        super().__init__(data)
        self.name = os.path.basename(path)
        self.path = path
        self.size = len(data)
        self.type = ""

    def getvalue(self) -> bytes:
        return self.getbuffer().tobytes()

    # Value-based identity so the object is stable across Streamlit reruns
    # (e.g. the plugin ``create_widget`` state-diff and caching keys).
    def __eq__(self, other) -> bool:
        return isinstance(other, LocalFile) and other.path == self.path

    def __hash__(self) -> int:
        return hash(self.path)

    def __reduce__(self):
        return (self.__class__, (self.path,))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"LocalFile(name={self.name!r}, path={self.path!r})"


def _scan_folder(folder: str, exts: list[str] | None) -> list[str]:
    """Return sorted file paths directly inside *folder* matching the extensions.

    Only the top level of *folder* is scanned (no recursion). A missing folder
    is created so the expected data location always exists for the user to fill.
    """
    if not folder:
        return []
    # Make sure the data directory exists; create it if it doesn't.
    os.makedirs(folder, exist_ok=True)
    out = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if (
            exts is None
            or os.path.splitext(path)[1].lower().lstrip(".") in exts
        ):
            out.append(path)
    return sorted(out)


@overload
def file_input(
    label: str,
    type: str | list[str] | None = ...,
    accept_multiple_files: Literal[False] = ...,
    key: str | None = ...,
    help: str | None = ...,
    *,
    default_dir: str = ...,
    default_source: str = ...,
    container=...,
) -> "LocalFile | None": ...


@overload
def file_input(
    label: str,
    type: str | list[str] | None,
    accept_multiple_files: Literal[True],
    key: str | None = ...,
    help: str | None = ...,
    *,
    default_dir: str = ...,
    default_source: str = ...,
    container=...,
) -> "list[LocalFile]": ...


def file_input(
    label: str,
    type: str | list[str] | None = None,
    accept_multiple_files: bool = False,
    key: str | None = None,
    help: str | None = None,
    *,
    default_dir: str = ".",
    default_source: str = "folder",
    container=None,
) -> "LocalFile | list[LocalFile] | None":
    """Drop-in replacement for ``st.file_uploader`` with a server-folder mode.

    A toggle switches between two sources:

    * **Server folder** (default): a text input chooses a directory to scan and
      a select/multiselect lists the matching files. Designed for running on a
      remote server where uploading over a slow connection is impractical.
    * **Browser upload**: the standard ``st.file_uploader``.

    Return value matches ``st.file_uploader``: a single file-like object (or
    ``None``) when ``accept_multiple_files`` is ``False``, otherwise a list.
    Folder selections are returned as :class:`LocalFile` instances.

    Args:
        label: Label shown on the file selection widget.
        type: Allowed extension(s) (string or list), e.g. ``"json"`` or
            ``["pickle", "pkl"]``. ``None`` allows any file.
        accept_multiple_files: Allow selecting more than one file.
        key: Unique key prefix (REQUIRED) for widget state isolation.
        help: Tooltip for the selection widget.
        default_dir: Folder pre-filled in the scan path input.
        default_source: ``"folder"`` (default) or ``"upload"``.
        container: Streamlit container to render into (e.g. ``st.sidebar``).
            Defaults to the top-level ``st``.
    """
    if key is None:
        raise ValueError(
            "file_input requires a unique 'key' to isolate widget state."
        )
    target = container if container is not None else st

    # Normalise the type filter to a list of bare lowercase extensions.
    exts: list[str] | None = None
    if type is not None:
        type_list = [type] if isinstance(type, str) else list(type)
        exts = [t.lower().lstrip(".") for t in type_list]

    # A default only where the key holds nothing yet: a caller may seed the
    # keys (a restored choice), and value= next to a seeded key is a
    # Streamlit warning on every run.
    def _default(sub: str, value):
        return {} if f"{key}{sub}" in st.session_state else {"value": value}

    use_upload = target.toggle(
        "📤 Upload from browser instead",
        **_default("__source_is_upload", default_source == "upload"),
        key=f"{key}__source_is_upload",
        help="Off: pick files already present in a folder on the server. "
        "On: upload from your local machine (slow over a poor connection).",
    )

    if use_upload:
        return cast(
            "LocalFile | list[LocalFile] | None",
            target.file_uploader(
                label,
                type=type,
                accept_multiple_files=accept_multiple_files,
                key=f"{key}__uploader",
                help=help,
            ),
        )

    # --- Server-folder mode ---
    folder = target.text_input(
        "📁 Server folder to scan",
        **_default("__folder", default_dir),
        key=f"{key}__folder",
        help="Path on the server to search for files (this folder only).",
    )

    matches = _scan_folder(folder, exts)
    if not matches:
        hint = f" matching {exts}" if exts else ""
        target.info(f"No files{hint} found in `{folder or '∅'}`.")
        return [] if accept_multiple_files else None

    def _rel(path: str) -> str:
        try:
            return os.path.relpath(path, folder)
        except ValueError:
            return path

    if accept_multiple_files:
        chosen = target.multiselect(
            label,
            options=matches,
            format_func=_rel,
            key=f"{key}__multiselect",
            help=help,
        )
        return [LocalFile(p) for p in chosen]

    chosen = target.selectbox(
        label,
        options=matches,
        format_func=_rel,
        index=None,
        placeholder="Select a file from the folder…",
        key=f"{key}__selectbox",
        help=help,
    )
    return LocalFile(chosen) if chosen else None


def generate_unique_filename(plugin_name, data, *args, **kwargs):
    """
    Generate a unique filename by combining the plugin_name with a hash of the serialized data.

    Parameters:
        plugin_name (str): The name of the plugin.
        data: The data to be serialized and hashed.

    Returns:
        str: A unique filename in the format "plugin_name_hash.pickle".
    """
    combined_data = (plugin_name, data, args, kwargs)
    serialized_data = pickle.dumps(combined_data)
    hash_object = hashlib.sha256(serialized_data)
    filename = f"{plugin_name}^" + hash_object.hexdigest()
    return filename


def save_to_pickle(data, filename, folder="cache"):
    """
    Save data to a pickle file and return the file path.

    Parameters:
        data: The data to be saved.
        filename (str): The name of the pickle file.
        folder (str): The folder where the pickle file will be saved.

    Returns:
        str: The full path to the saved pickle file.
    """
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Create the full file path
    file_path = os.path.join(folder, filename)  # type: ignore

    # Save the data to the pickle file
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    return file_path


def read_data(
    data, save_flag, shape=None, dtype: np.dtype | type = np.complex64
):
    if save_flag:
        if data.endswith(".pickle"):
            with open(data, "rb") as file:
                readed_data = pickle.load(file)
        elif data.endswith(".bin"):
            readed_data = np.memmap(
                data,
                dtype=dtype,
                mode="r",
                shape=shape,
            )
        else:
            raise ValueError(
                f"Unsupported file extension for '{data}'; expected .pickle or .bin"
            )
    else:
        readed_data = data  # 7D np.ndarray
    return readed_data


# Helper function to safely parse values, especially for None and numbers
def safe_literal_eval(value_str, expected_type=None, allow_none=False):
    """
    Safely evaluate a string literal, handling None and basic types.

    Parameters:
        value_str (str): The string to be evaluated. It can represent a literal value like a number, string, or None.
        expected_type (str, optional): The expected type of the evaluated value.
            Can be "int", "float", or "str". If provided, the function will enforce type checking.
        allow_none (bool, optional): Whether to allow the string "None" to be evaluated as None.
            If False, a ValueError will be raised if "None" is encountered.

    Returns:
        The evaluated value, which can be an int, float, str, list, or None, depending on the input.

    Raises:
        ValueError: If the input string cannot be evaluated, or if the evaluated value does not match the expected type,
                or if "None" is encountered but `allow_none` is False.
    """
    # self.logger.debug(f"Attempting safe_literal_eval on '{value_str}' (expected: {expected_type}, allow_none: {allow_none})") # Cannot log here as it's a global function
    try:
        # Handle direct None string
        if isinstance(value_str, str) and value_str.strip().lower() == "none":
            if allow_none:
                # self.logger.debug("Evaluated 'None' string as None.")
                return None
            else:
                # self.logger.warning(f"Disallowed 'None' string encountered for value '{value_str}'.")
                raise ValueError("None is not allowed for this parameter.")

        # Evaluate other literals using ast.literal_eval, which safely evaluates strings to Python literals
        val = ast.literal_eval(value_str)
        # self.logger.debug(f"ast.literal_eval result: {val} (type: {type(val)})")

        # Type checking for single values. A ``list`` result means the string
        # was a list literal, which is validated element-wise by the caller.
        if (
            expected_type == "int"
            and not isinstance(val, int)
            and not isinstance(val, list)
        ):
            raise TypeError(f"Expected an integer, got {type(val)}")
        # numbers.Number lets an int stand in for a float
        if (
            expected_type == "float"
            and not isinstance(val, numbers.Number)
            and not isinstance(val, list)
        ):
            raise TypeError(f"Expected a float, got {type(val)}")
        if (
            expected_type == "str"
            and not isinstance(val, str)
            and not isinstance(val, list)
        ):
            raise TypeError(f"Expected a string, got {type(val)}")

        # Check for None if not allowed (after evaluation)
        if val is None and not allow_none:
            # self.logger.warning(f"Disallowed None value encountered after evaluation for '{value_str}'.")
            raise ValueError("None is not allowed for this parameter.")

        # self.logger.debug(f"Successfully evaluated '{value_str}' to: {val}")
        return val
    except (ValueError, SyntaxError, TypeError) as e:
        # self.logger.error(f"Evaluation failed for '{value_str}': {e}")
        raise ValueError(f"Invalid input format '{value_str}': {e}")
