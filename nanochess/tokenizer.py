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


class MoveTokenizer:
    """Move-level tokenizer over the fixed UCI vocabulary."""

    name = "move"

    def __init__(self) -> None:
        self.moves = build_move_vocab()
        self.id_to_token: list[str] = SPECIALS + self.moves
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(self.id_to_token)}
        self._special_ids = {self.token_to_id[t] for t in SPECIALS}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

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

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"name": self.name, "tokens": self.id_to_token}
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> MoveTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("name") != cls.name:
            raise ValueError(f"not a {cls.name} tokenizer: {payload.get('name')!r}")
        tok = cls.__new__(cls)
        tok.id_to_token = list(payload["tokens"])
        tok.moves = [t for t in tok.id_to_token if t not in SPECIALS]
        tok.token_to_id = {t: i for i, t in enumerate(tok.id_to_token)}
        tok._special_ids = {tok.token_to_id[t] for t in SPECIALS}
        return tok
