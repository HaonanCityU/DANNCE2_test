"""Data loading and saving operations."""
import numpy as np
import scipy.io as sio
from typing import List, Dict, Text, Union
import mat73


def load_label3d_data(path: Text, key: Text):
    """Load Label3D data

    Args:
        path (Text): Path to Label3D file
        key (Text): Field to access

    Returns:
        TYPE: Data from field
    """
    # First try scipy.io.loadmat (for v7 and earlier MATLAB files)
    try: 
        mat_file = sio.loadmat(path)
        if key not in mat_file:
            available_keys = [k for k in mat_file.keys() if not k.startswith('__')]
            raise KeyError(
                f"Key '{key}' not found in {path}. Available keys: {available_keys}"
            )
        d = mat_file[key]
        dataset = [f[0] for f in d]

        # Data are loaded in this annoying structure where the array
        # we want is at dataset[i][key][0,0], as a nested array of arrays.
        # Simplify this structure (a numpy record array) here.
        # Additionally, cannot use views here because of shape mismatches. Define
        # new dict and return.
        data = []
        for d in dataset:
            d_ = {}
            for key in d.dtype.names:
                d_[key] = d[key][0, 0]
            data.append(d_)
    except KeyError:
        # Re-raise KeyError with better message (already handled above)
        raise
    except Exception as e:
        # If scipy.io.loadmat fails (e.g., file is v7.3 format), try mat73
        try:
            mat_file = mat73.loadmat(path)
            if key not in mat_file:
                available_keys = [k for k in mat_file.keys() if not k.startswith('__')]
                raise KeyError(
                    f"Key '{key}' not found in {path} (v7.3 format). Available keys: {available_keys}"
                )
            d = mat_file[key]
            data = [f[0] for f in d]
        except TypeError as te:
            # mat73 raises TypeError if file is not v7.3 format
            raise TypeError(
                f"Failed to load {path}. File is not MATLAB v7.3 format, and scipy.io.loadmat also failed. "
                f"Original error: {str(e)}. mat73 error: {str(te)}"
            ) from te
        except KeyError:
            # Re-raise KeyError from mat73
            raise
        except Exception as e2:
            # If mat73 also fails for other reasons, raise with context
            raise RuntimeError(
                f"Failed to load {path} with both scipy.io.loadmat and mat73. "
                f"scipy error: {str(e)}. mat73 error: {str(e2)}"
            ) from e2
    return data


def load_camera_params(path: Text) -> List[Dict]:
    """Load camera parameters from Label3D file.

    Args:
        path (Text): Path to Label3D file

    Returns:
        List[Dict]: List of camera parameter dictionaries.
    """
    params = load_label3d_data(path, "params")
    for p in params:
        if "r" in p:
            p["R"] = p["r"]
    return params


def load_sync(path: Text) -> List[Dict]:
    """Load synchronization data from Label3D file.

    Args:
        path (Text): Path to Label3D file.

    Returns:
        List[Dict]: List of synchronization dictionaries.
    """
    dataset = load_label3d_data(path, "sync")
    for d in dataset:
        d["data_frame"] = d["data_frame"].astype(int)
        d["data_sampleID"] = d["data_sampleID"].astype(int)
    return dataset


def load_labels(path: Text) -> List[Dict]:
    """Load labelData from Label3D file.

    Args:
        path (Text): Path to Label3D file.

    Returns:
        List[Dict]: List of labelData dictionaries.
    """
    dataset = load_label3d_data(path, "labelData")
    for d in dataset:
        d["data_frame"] = d["data_frame"].astype(int)
        d["data_sampleID"] = d["data_sampleID"].astype(int)
    return dataset


def load_com(path: Text) -> Dict:
    """Load COM from .mat file.

    Args:
        path (Text): Path to .mat file with "com" field

    Returns:
        Dict: Dictionary with com data
    """
    def _maybe_get(root, *keys):
        for k in keys:
            if isinstance(root, dict) and k in root:
                return root[k]
        return None

    def _extract_struct_field(x, field: str):
        """
        Support both:
        - scipy.io.loadmat struct arrays (np.ndarray with dtype.names)
        - mat73 dicts (plain dict)
        """
        if isinstance(x, dict):
            return x.get(field, None)
        # scipy struct: typically shape (1,1) ndarray with dtype.names
        if hasattr(x, "dtype") and getattr(x.dtype, "names", None) and field in x.dtype.names:
            try:
                return x[field]
            except Exception:
                return None
        return None

    # Load mat (v7.x via scipy; v7.3 via mat73)
    try:
        root = sio.loadmat(path)
    except Exception:
        root = mat73.loadmat(path)

    # Common layouts:
    # 1) root["com"] is a struct/dict with fields com3d, sampleID
    # 2) root has top-level keys com3d, sampleID (no "com" wrapper)
    # 3) root has top-level keys com (numeric Nx3 or Nx3xK) and sampleID
    com_obj = _maybe_get(root, "com", "COM", "comData", "comdata")

    com3d = None
    sampleID = None

    if com_obj is not None:
        com3d = _extract_struct_field(com_obj, "com3d")
        sampleID = _extract_struct_field(com_obj, "sampleID")
        # scipy struct often needs [0,0] unwrapping
        if isinstance(com3d, np.ndarray) and com3d.shape == (1, 1):
            com3d = com3d[0, 0]
        if isinstance(sampleID, np.ndarray) and sampleID.shape == (1, 1):
            sampleID = sampleID[0, 0]
        # If "com" is a numeric array (common for com3d*.mat outputs in this repo),
        # treat it as com3d.
        if com3d is None and isinstance(com_obj, np.ndarray) and com_obj.dtype.names is None:
            com3d = com_obj

    if com3d is None:
        com3d = _maybe_get(root, "com3d", "COM3D", "com_3d", "COM_3D")
    if sampleID is None:
        sampleID = _maybe_get(root, "sampleID", "sampleId", "SampleID", "SAMPLEID")

    if com3d is None or sampleID is None:
        available = sorted(list(root.keys())) if isinstance(root, dict) else []
        raise KeyError(
            f"COM .mat must contain either a top-level 'com' struct with fields "
            f"'com3d' and 'sampleID', or top-level keys 'com3d' and 'sampleID'. "
            f"Got keys: {available}"
        )

    data = {}
    data["com3d"] = com3d
    # sampleID can come in as (N,), (1,N), (N,1), or even object-wrapped arrays
    sid = np.asarray(sampleID)
    # Unwrap common scipy "cell-like" object arrays: shape (1, N) with dtype=object
    if sid.dtype == object and sid.size == 1:
        sid = np.asarray(sid.item())
    sid = np.asarray(sid).astype(int).reshape(-1)
    data["sampleID"] = sid.reshape(1, -1)
    return data


def load_camnames(path: Text) -> Union[List, None]:
    """Load camera names from .mat file.

    Args:
        path (Text): Path to .mat file with "camnames" field

    Returns:
        Union[List, None]: List of cameranames
    """
    def _to_py_str(x) -> str:
        """Robustly convert scipy/mat73-loaded MATLAB strings to Python str."""
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, (bytes, bytearray)):
            try:
                return x.decode("utf-8")
            except Exception:
                return x.decode(errors="ignore")
        if isinstance(x, np.ndarray):
            # Unwrap scalar arrays
            if x.size == 1:
                try:
                    return _to_py_str(x.item())
                except Exception:
                    pass
            # MATLAB char arrays often come in as (1, N) arrays of single characters
            if x.dtype.kind in {"U", "S"}:
                # If it's an array of single characters, join them
                flat = x.reshape(-1)
                if flat.size > 1 and all(isinstance(c, (str, bytes, np.str_, np.bytes_)) for c in flat):
                    try:
                        return "".join([c.decode("utf-8") if isinstance(c, (bytes, bytearray, np.bytes_)) else str(c) for c in flat]).strip()
                    except Exception:
                        return "".join([str(c) for c in flat]).strip()
                # Otherwise, best-effort stringify
                try:
                    return str(x)
                except Exception:
                    return ""
            # Object arrays (cell arrays) – try first element
            if x.dtype == object and x.size >= 1:
                try:
                    return _to_py_str(x.flat[0])
                except Exception:
                    return ""
        # Fallback
        return str(x)

    try:
        label_3d_file = sio.loadmat(path)
        if "camnames" in label_3d_file:
            names = label_3d_file["camnames"]
            # Flatten common MATLAB shapes: (1, N), (N, 1), or (N,)
            flat = np.asarray(names).reshape(-1)
            camnames = [_to_py_str(n).strip() for n in flat]
            camnames = [c for c in camnames if c != ""]
        else:
            camnames = None
    except Exception:
        label_3d_file = mat73.loadmat(path)
        if "camnames" in label_3d_file:
            names = label_3d_file["camnames"]
            # mat73 typically yields Python lists/strings already, but be defensive.
            if isinstance(names, list):
                camnames = [_to_py_str(n).strip() for n in names]
            else:
                camnames = [_to_py_str(n).strip() for n in np.asarray(names).reshape(-1)]
            camnames = [c for c in camnames if c != ""]
        else:
            camnames = None
    
    return camnames
