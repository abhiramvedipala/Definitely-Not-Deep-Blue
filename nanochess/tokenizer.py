"""Tokenizers for chess move sequences.

The primary tokenizer is move-level over UCI notation: one token per ply. This
is the decision that makes the eval clean — because one token is exactly one
move, "the model emitted an illegal move" is a well-defined statement about a
single token, with no parsing layer in between.

The vocabulary is *generated*, never derived from a corpus, so it does not
change depending on which month of Lichess games you happened to download.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]

_PROMOTION_PIECES = "bnqr"  # sorted, so the vocabulary is deterministic


def build_move_vocab() -> list[str]:
    """Enumerate every geometrically possible UCI move.

    A move is possible if some piece could make it on an otherwise empty board.
    Placing a lone queen on each origin square covers all sliding moves, all
    king moves and both castles (in UCI, castling is just ``e1g1``); a lone
    knight covers the leaps. Promotions are added separately because they carry
    a suffix the origin/destination enumeration cannot express.
    """
    moves: set[str] = set()

    for origin in chess.SQUARES:
        for piece_type in (chess.QUEEN, chess.KNIGHT):
            board = chess.Board(None)
            board.set_piece_at(origin, chess.Piece(piece_type, chess.WHITE))
            board.turn = chess.WHITE
            for move in board.legal_moves:
                moves.add(move.uci())

    # Promotions: a pawn on the 7th (white) or 2nd (black) rank stepping
    # forward or capturing diagonally onto the back rank.
    for from_file in range(8):
        for to_file in range(max(0, from_file - 1), min(8, from_file + 2)):
            for piece in _PROMOTION_PIECES:
                white = chess.square_name(chess.square(from_file, 6)) + chess.square_name(
                    chess.square(to_file, 7)
                )
                black = chess.square_name(chess.square(from_file, 1)) + chess.square_name(
                    chess.square(to_file, 0)
                )
                moves.add(white + piece)
                moves.add(black + piece)

    return sorted(moves)


class _BaseTokenizer:
    """Shared vocabulary bookkeeping and on-disk format."""

    name = "base"

    def _init_vocab(self, tokens: list[str]) -> None:
        self.id_to_token: list[str] = tokens
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(tokens)}
        self._special_ids = {self.token_to_id[t] for t in SPECIALS}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def moves(self) -> list[str]:
        """The non-special vocabulary. Derived, not stored, so a tokenizer
        restored from disk (which bypasses __init__) has it too."""
        return [t for t in self.id_to_token if t not in SPECIALS]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"name": self.name, "tokens": self.id_to_token}
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("name") != cls.name:
            raise ValueError(f"not a {cls.name} tokenizer: {payload.get('name')!r}")
        tok = cls.__new__(cls)
        tok._init_vocab(list(payload["tokens"]))
        return tok


class MoveTokenizer(_BaseTokenizer):
    """Move-level tokenizer over the fixed UCI vocabulary. One token per ply."""

    name = "move"

    def __init__(self) -> None:
        self._init_vocab(SPECIALS + build_move_vocab())

    def encode(self, moves: list[str], add_special: bool = False) -> list[int]:
        ids = [self.token_to_id.get(m, self.unk_id) for m in moves]
        if add_special:
            ids = [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int], keep_special: bool = False) -> list[str]:
        out = []
        for i in ids:
            if not keep_special and i in self._special_ids:
                continue
            out.append(self.id_to_token[i])
        return out


class CharTokenizer(_BaseTokenizer):
    """Character-level tokenizer over UCI text, with a space between moves.

    Kept as a genuine second implementation rather than a curiosity: it proves
    the tokenizer abstraction is real, and it makes the cost of the choice
    measurable — the same game costs ~5x more tokens here, which is exactly why
    the move-level tokenizer wins for a 256-token context.
    """

    name = "char"
    SEP = " "

    def __init__(self) -> None:
        chars = [self.SEP, *sorted("abcdefgh12345678" + _PROMOTION_PIECES)]
        self._init_vocab(SPECIALS + chars)

    def encode(self, moves: list[str], add_special: bool = False) -> list[int]:
        text = self.SEP.join(moves)
        ids = [self.token_to_id.get(c, self.unk_id) for c in text]
        if add_special:
            ids = [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int], keep_special: bool = False) -> list[str]:
        chars = [
            self.id_to_token[i]
            for i in ids
            if keep_special or i not in self._special_ids
        ]
        text = "".join(chars).strip()
        return text.split(self.SEP) if text else []


_REGISTRY = {cls.name: cls for cls in (MoveTokenizer, CharTokenizer)}


def load_tokenizer(path: str | Path) -> _BaseTokenizer:
    """Load whichever tokenizer was saved to ``path``, dispatching on its name."""
    name = json.loads(Path(path).read_text(encoding="utf-8")).get("name")
    if name not in _REGISTRY:
        raise ValueError(f"unknown tokenizer {name!r}")
    return _REGISTRY[name].load(path)
