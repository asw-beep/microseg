"""Multi-channel microscopy image container.

Everything downstream (preprocessing, segmentation, feature extraction) operates
on :class:`MicroscopyImage`, which always carries an ``(H, W, C)`` float32 array
plus per-channel metadata.  BBBC038 is effectively single-channel nuclei data,
but real assays are multiplexed (DAPI + phalloidin + a marker; 4-6 channels in
high-content screening), so the container is channel-aware from the start:

* every channel has a ``name`` and a biological ``role``;
* preprocessing ops take a ``channels=`` selector and are applied per channel;
* segmentation reads the ``nuclei`` role channel, while feature extraction
  quantifies intensity and texture in *every* channel using the same labels.

That last point is why the abstraction exists: in practice you segment on one
channel and measure on all of them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

# Biological roles we know how to reason about.  ``marker`` is the catch-all for
# a stain that is only ever measured, never segmented on.
KNOWN_ROLES = ("nuclei", "cytoplasm", "membrane", "marker", "brightfield")


@dataclass(frozen=True)
class ChannelSpec:
    """Metadata for one channel of a microscopy image."""

    name: str
    role: str = "marker"
    wavelength_nm: float | None = None

    def __post_init__(self) -> None:
        if self.role not in KNOWN_ROLES:
            raise ValueError(
                f"unknown channel role {self.role!r}; expected one of {KNOWN_ROLES}"
            )


@dataclass
class MicroscopyImage:
    """An ``(H, W, C)`` float32 image in ``[0, 1]`` with channel metadata.

    Parameters
    ----------
    data:
        Pixel data.  2-D input is promoted to a single channel.
    channels:
        One :class:`ChannelSpec` per channel.
    pixel_size_um:
        Physical pixel size.  When known, area and perimeter features are
        reported in microns alongside pixels.
    name:
        Identifier used for output filenames and result tables.
    """

    data: np.ndarray
    channels: list[ChannelSpec]
    pixel_size_um: float | None = None
    name: str = "image"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.data)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise ValueError(f"expected 2-D or 3-D image data, got shape {arr.shape}")
        self.data = arr.astype(np.float32, copy=False)
        if len(self.channels) != self.data.shape[2]:
            raise ValueError(
                f"{len(self.channels)} channel specs for {self.data.shape[2]} channels"
            )

    # ------------------------------------------------------------------ shape
    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[2])

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.n_channels)

    @property
    def channel_names(self) -> list[str]:
        return [c.name for c in self.channels]

    # --------------------------------------------------------------- indexing
    def index_of(self, key: str) -> int:
        """Resolve a channel name (preferred) or role to a channel index."""
        for i, spec in enumerate(self.channels):
            if spec.name == key:
                return i
        for i, spec in enumerate(self.channels):
            if spec.role == key:
                return i
        raise KeyError(
            f"no channel named or with role {key!r}; have "
            f"{[(c.name, c.role) for c in self.channels]}"
        )

    def resolve(self, channels: str | int | Sequence | None) -> list[int]:
        """Normalise a channel selector into a list of indices.

        ``None`` selects every channel, which is what preprocessing ops default
        to so a 5-channel plate needs no special-casing.
        """
        if channels is None:
            return list(range(self.n_channels))
        if isinstance(channels, (str, int, np.integer)):
            channels = [channels]
        out: list[int] = []
        for key in channels:
            if isinstance(key, (int, np.integer)):
                out.append(int(key))
            else:
                out.append(self.index_of(key))
        return out

    def channel(self, key: str | int) -> np.ndarray:
        """Return one channel as a 2-D array (a view, not a copy)."""
        idx = int(key) if isinstance(key, (int, np.integer)) else self.index_of(key)
        return self.data[:, :, idx]

    def nuclei(self) -> np.ndarray:
        """The channel segmentation runs on: role ``nuclei``, else channel 0."""
        try:
            return self.channel(self.index_of("nuclei"))
        except KeyError:
            return self.data[:, :, 0]

    def with_data(self, data: np.ndarray, **overrides) -> MicroscopyImage:
        """Copy carrying new pixel data, preserving metadata."""
        return replace(self, data=data, **overrides)

    # ---------------------------------------------------------------- exports
    def to_uint8(self, key: str | int | None = None) -> np.ndarray:
        """Display-ready uint8 array (one channel, or all of them)."""
        arr = self.data if key is None else self.channel(key)
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    def to_rgb(self) -> np.ndarray:
        """Best-effort RGB view for visualisation.

        1 channel -> grey, 3 channels -> as-is, otherwise the first three are
        mapped to R/G/B and the rest dropped.  Beyond three channels there is no
        faithful RGB, so this is explicitly a preview rather than a conversion.
        """
        arr = np.clip(self.data, 0.0, 1.0)
        if self.n_channels == 1:
            return np.repeat(arr, 3, axis=2)
        if self.n_channels == 3:
            return arr
        rgb = np.zeros((self.height, self.width, 3), np.float32)
        k = min(3, self.n_channels)
        rgb[:, :, :k] = arr[:, :, :k]
        return rgb

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        channels: Iterable | None = None,
        *,
        name: str = "image",
        pixel_size_um: float | None = None,
        scale_dtype: bool = True,
    ) -> MicroscopyImage:
        """Build from a raw array, scaling integer dtypes into ``[0, 1]``.

        ``scale_dtype`` only divides by the dtype range (uint8 -> /255, uint16 ->
        /65535).  It is *not* intensity normalisation, which lives in
        :mod:`microseg.preprocessing.normalize` and is a deliberate pipeline step
        rather than a silent side effect of loading.
        """
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if scale_dtype and np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32, copy=False)

        if channels is None:
            specs = default_specs(arr.shape[2])
        else:
            specs = [
                c if isinstance(c, ChannelSpec) else ChannelSpec(str(c)) for c in channels
            ]
        return cls(arr, specs, pixel_size_um=pixel_size_um, name=name)


def default_specs(n: int) -> list[ChannelSpec]:
    """Sensible channel specs when the caller supplies none."""
    if n == 1:
        return [ChannelSpec("ch0", "nuclei")]
    if n == 3:
        # An RGB micrograph.  The nuclear signal is recovered by the loader, so
        # here we only label the axes.
        return [ChannelSpec("r"), ChannelSpec("g"), ChannelSpec("b")]
    return [ChannelSpec(f"ch{i}") for i in range(n)]
