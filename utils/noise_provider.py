# utils/noise_provider.py
import os
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import scipy.io as sio

NOISE_FILE_CANDIDATES = [
    "additive.mat", "deadlines.mat", "jpeg.mat", "kernal.mat",
    "poisson.mat", "salt_pepper.mat", "stripes.mat", "zmguass.mat",
]

PREFERRED_VAR_KEYS = [
    "noise", "img", "image", "cube", "data", "X", "Y", "Z",
    "pavia", "paviaU", "PaviaU", "noisy", "noised", "degraded"
]

def _is_numeric_ndarray(x: Any) -> bool:
    return isinstance(x, np.ndarray) and x.dtype != object

def _first_non_dunder_key(d: dict) -> Optional[str]:
    for k in d.keys():
        if not k.startswith("__"):
            return k
    return None

def _unwrap_mat_obj(x: Any) -> Any:
    """
    递归剥离 MATLAB cell/struct：
    - cell：ndarray(dtype=object)，取 item() 继续
    - struct：有 _fieldnames / dtype.names，取第一个字段继续
    - list/tuple：取第一个非空元素继续
    """
    # 循环直到得到“数值型ndarray”或无法再剥
    visited = set()
    while True:
        if _is_numeric_ndarray(x):
            return x
        # object ndarray（cell / struct容器）
        if isinstance(x, np.ndarray) and x.dtype == object:
            x = np.squeeze(x)
            # 空直接报错
            if x.size == 0:
                raise RuntimeError("MAT对象为空（object ndarray size=0）")
            # 取单元素
            if x.size == 1:
                x = x.item()
                continue
            # 多元素cell，尝试找第一个能继续剥的
            for i in range(x.size):
                try:
                    cand = x.flat[i]
                    y = _unwrap_mat_obj(cand)
                    if _is_numeric_ndarray(y):
                        return y
                except Exception:
                    continue
            # 如果都不行，抛错
            raise RuntimeError("未能从多元素 object ndarray 中解析到数值数组")
        # scipy 的 mat_struct（旧接口）
        if hasattr(x, "_fieldnames"):
            fields = list(getattr(x, "_fieldnames") or [])
            if not fields:
                raise RuntimeError("mat_struct 无字段")
            # 依次尝试优先字段
            pref = [f for f in PREFERRED_VAR_KEYS if f in fields] + fields
            for f in pref:
                y = getattr(x, f)
                try:
                    y = _unwrap_mat_obj(y)
                    if _is_numeric_ndarray(y):
                        return y
                except Exception:
                    continue
            raise RuntimeError("未能从 mat_struct 中解析到数值数组")
        # 结构化数组（dtype.names）
        if isinstance(x, np.void) or (isinstance(x, np.ndarray) and x.dtype.names):
            names = list(x.dtype.names or [])
            pref = [f for f in PREFERRED_VAR_KEYS if f in names] + names
            for f in pref:
                try:
                    y = x[f]
                    y = _unwrap_mat_obj(y)
                    if _is_numeric_ndarray(y):
                        return y
                except Exception:
                    continue
            raise RuntimeError("未能从结构化数组中解析到数值数组")
        # list / tuple
        if isinstance(x, (list, tuple)):
            for el in x:
                try:
                    y = _unwrap_mat_obj(el)
                    if _is_numeric_ndarray(y):
                        return y
                except Exception:
                    continue
            raise RuntimeError("未能从 list/tuple 中解析到数值数组")
        # 防止循环
        obj_id = id(x)
        if obj_id in visited:
            raise RuntimeError("解析 .mat 过程中检测到循环引用")
        visited.add(obj_id)
        # 最后兜底：直接返回（让上层处理错误）
        return x

def _pick_primary_array(mat_dict: dict, key_fallback: Optional[str]) -> np.ndarray:
    # 先按偏好 key
    if key_fallback and key_fallback in mat_dict:
        arr = _unwrap_mat_obj(mat_dict[key_fallback])
        if _is_numeric_ndarray(arr):
            return arr
    for k in PREFERRED_VAR_KEYS:
        if k in mat_dict:
            arr = _unwrap_mat_obj(mat_dict[k])
            if _is_numeric_ndarray(arr):
                return arr
    # 再选第一个非 __ 的键
    k0 = _first_non_dunder_key(mat_dict)
    if k0 is not None:
        arr = _unwrap_mat_obj(mat_dict[k0])
        if _is_numeric_ndarray(arr):
            return arr
    # 遍历 value 尝试
    for v in mat_dict.values():
        try:
            arr = _unwrap_mat_obj(v)
            if _is_numeric_ndarray(arr):
                return arr
        except Exception:
            continue
    raise RuntimeError("无法在 .mat 文件中解析出数值数组")

def _to_hwc(arr: np.ndarray) -> np.ndarray:
    # squeeze 掉多余维
    arr = np.array(arr)
    while arr.ndim > 3:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim == 2:
        arr = arr[:, :, None]  # (H,W,1)
    elif arr.ndim == 3:
        # 若是 (C,H,W) 形式（常见：C 小，且 H、W 大）
        H, W, C = arr.shape
        if H < 16 and C > 16:
            # (C,H,W) -> (H,W,C)
            arr = np.transpose(arr, (1, 2, 0))
        # 若是 (H,W,C) 则保持
    else:
        raise RuntimeError(f"异常的数组维度: {arr.shape}")
    # 保证连续/float32
    arr = np.ascontiguousarray(arr).astype(np.float32)
    return arr

class NoiseProvider:
    """
    读取 noise_Pavia/*.mat ，并在训练时根据 (row, col, patch_size)
    抠出与干净样本“同位置”的退化patch。
    自动适配多种 MATLAB 保存格式：cell、struct、(C,H,W)/(H,W,C) 等。
    """
    def __init__(self, noise_root: str, patch_size: int, key_fallback: Optional[str] = None):
        self.noise_root = noise_root
        self.ps = patch_size
        self.key_fallback = key_fallback
        self._cubes: Dict[str, np.ndarray] = {}   # noise_name -> (H, W, C)
        self._load_all()

    def _load_one(self, path: str) -> Tuple[str, np.ndarray]:
        mdict = sio.loadmat(path, struct_as_record=False, squeeze_me=False)
        arr = _pick_primary_array(mdict, self.key_fallback)
        arr = _to_hwc(arr)  # -> (H,W,C)
        key = os.path.splitext(os.path.basename(path))[0]
        return key, arr

    def _load_all(self):
        files = [f for f in os.listdir(self.noise_root) if f.endswith(".mat")]
        if not files:
            raise FileNotFoundError(f"在 {self.noise_root} 下未发现任何 .mat 噪声文件")
        for fname in files:
            fpath = os.path.join(self.noise_root, fname)
            key, cube = self._load_one(fpath)
            self._cubes[key] = cube
        # 尺寸一致性检测
        shapes = {v.shape for v in self._cubes.values()}
        if len(shapes) > 1:
            raise RuntimeError(f"噪声文件尺寸不一致: {shapes}")
        self._ref_shape = next(iter(self._cubes.values())).shape  # (H,W,C)

    @property
    def noise_types(self) -> List[str]:
        return sorted(list(self._cubes.keys()))

    def sample_noise_types(self, n: int, per_batch: bool = False) -> List[str]:
        import random
        types = self.noise_types
        if per_batch:
            t = random.choice(types)
            return [t] * n
        return [random.choice(types) for _ in range(n)]

    def get_patch(self, noise_type: str, row: int, col: int, ps: Optional[int] = None) -> np.ndarray:
        if ps is None:
            ps = self.ps
        cube = self._cubes[noise_type]  # (H,W,C)
        H, W, C = cube.shape
        half = ps // 2

        r0, r1 = row - half, row + half + 1
        c0, c1 = col - half, col + half + 1

        rr = np.clip(np.arange(r0, r1), 0, H - 1)
        cc = np.clip(np.arange(c0, c1), 0, W - 1)

        patch = cube[rr][:, cc]           # (ps, ps, C)
        patch = np.transpose(patch, (2, 0, 1))  # (C, ps, ps)
        return patch.copy()

    def get_patches_batch(self, rows: np.ndarray, cols: np.ndarray, types: List[str], ps: Optional[int] = None) -> np.ndarray:
        assert len(rows) == len(cols) == len(types)
        patches = [self.get_patch(t, int(r), int(c), ps) for r, c, t in zip(rows, cols, types)]
        return np.stack(patches, axis=0)  # (B, C, ps, ps)
