from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class Emb:
    """Base class for managing embedders"""

    dimensions: Optional[int] = 1536

    def get_embedding(self, text: str) -> List[float]:
        raise NotImplementedError

    def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        embedding = self.get_embedding(text)
        return embedding, None
