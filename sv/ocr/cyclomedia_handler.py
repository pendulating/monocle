"""Old path. The handler is ``dagspaces.sources.cyclomedia.ocr:CyclomediaHandler``.

The OCR configs in sv/conf/ocr/data/ name the new path. This module keeps old
configs and old imports working.
"""

from dagspaces.sources.cyclomedia.ocr import CyclomediaHandler

__all__ = ["CyclomediaHandler"]
