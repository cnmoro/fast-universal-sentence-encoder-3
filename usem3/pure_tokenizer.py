"""Pure-python port of the sentencepiece normalizer + tokenizer.

This replaces the ``sentencepiece`` C++ dependency for the USE multilingual v3
model. It implements:

1. The ``nmt_nfkc`` normalizer from the precompiled charsmap: a darts-clone
   double-array trie + null-terminated replacement table, traversed exactly like
   sentencepiece's ``Normalizer::NormalizePrefix``.
2. Whitespace handling (add_dummy_prefix, escape_whitespaces -> U+2581,
   remove_extra_whitespaces).
3. Unigram Viterbi segmentation.

The USE multilingual v3 ``.model`` uses a *legacy* sentencepiece layout: the
ModelProto contains (field 1) the repeated pieces inline, (field 2) the
normalizer_spec, (field 3) the trainer_spec (which carries the precompiled
charsmap in field 2).

The vocabulary is held as one flat ``prefix -> piece id`` dict instead of a
nested per-character trie: same single dict lookup per character while walking a
position, but ~40 MB less resident memory for the 128k-piece vocabulary.
"""

import struct
from array import array
from collections import defaultdict

_UNK = 0
_BOS = 1
_EOS = 2


def _read_varint(data, pos):
    result = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _parse_fields(data, start, end):
    """Minimal protobuf reader returning {field: [(value_start, length)]}."""
    fields = defaultdict(list)
    pos = start
    while pos < end:
        b = data[pos]
        field = b >> 3
        wire = b & 7
        pos += 1
        if field == 0:  # multi-byte varint tag; skip
            while data[pos] & 0x80:
                pos += 1
            pos += 1
        elif wire == 0:  # varint
            while data[pos] & 0x80:
                pos += 1
            pos += 1
        elif wire == 1:  # 64-bit
            pos += 8
        elif wire == 2:  # length-delimited
            length, pos = _read_varint(data, pos)
            fields[field].append((pos, length))
            pos += length
        elif wire == 5:  # 32-bit
            fields[field].append((pos, 4))
            pos += 4
        else:
            break
    return fields


class _Normalizer:
    """nmt_nfkc normalizer over the precompiled darts-clone charsmap.

    The double-array walk is inlined (rather than split into helpers returning
    every common prefix) because it runs once per input byte. ASCII bytes take a
    fast path: very few charsmap keys start with an ASCII byte and continue (87
    of them in the bundled model), so one set lookup on the next byte tells us
    whether the walk could match more than the byte itself; when it cannot, the
    replacement comes straight from a 128-entry table.
    """

    def __init__(self, charsmap, add_dummy_prefix=True,
                 remove_extra_whitespaces=True, escape_whitespaces=True):
        if charsmap:
            trie_size = struct.unpack("<I", charsmap[:4])[0]
            trie = charsmap[4:4 + trie_size]
            self._units = struct.unpack(f"<{trie_size // 4}I", trie)
            self._normalized = charsmap[4 + trie_size:]
        else:
            self._units = None
            self._normalized = b""
        self._add_dummy_prefix = add_dummy_prefix
        self._remove_extra_whitespaces = remove_extra_whitespaces
        self._space_symbol = b"\xe2\x96\x81" if escape_whitespaces else b" "
        self._repl = {}  # trie value -> replacement bytes
        self._ascii, self._two_byte = self._build_ascii_table()

    def _build_ascii_table(self):
        """One-byte replacements for ASCII, plus the 2-byte keys of the trie.

        Returns (table, two_byte) where table[b] is what data[pos:] normalizes
        to when only the single byte b can match, and two_byte holds
        ``b << 8 | c`` for every 2-byte trie key, i.e. exactly the cases where
        the full walk has to run.
        """
        table = [bytes((b,)) for b in range(128)]
        units = self._units
        if units is None:
            return table, frozenset()
        two_byte = set()
        root = self._offset(units[0])
        for b in range(128):
            node_pos = root ^ b
            unit = units[node_pos]
            if (unit & 0x800000FF) != b:  # no trie key starts with this byte
                continue
            child = node_pos ^ self._offset(unit)
            if unit & 0x100:  # has_leaf: b alone normalizes to a replacement
                value = units[child] & 0x7FFFFFFF
                if value < len(self._normalized):
                    table[b] = self._replacement(value)
            for c in range(256):
                if (units[child ^ c] & 0x800000FF) == c:
                    two_byte.add((b << 8) | c)
        return table, frozenset(two_byte)

    @staticmethod
    def _offset(unit):
        return (unit >> 10) << ((unit & 0x200) >> 6)

    def _replacement(self, value):
        r = self._repl.get(value)
        if r is None:
            end = self._normalized.index(b"\x00", value)
            r = self._normalized[value:end]
            self._repl[value] = r
        return r

    def normalize_prefix(self, data, pos):
        """Return (normalized_prefix_bytes, consumed_byte_len) at data[pos:]."""
        n = len(data)
        if pos >= n:
            return b"", 0
        best_len = 0
        best_value = 0
        units = self._units
        if units is not None:
            unit = units[0]
            node_pos = (unit >> 10) << ((unit & 0x200) >> 6)
            i = pos
            while i < n:
                c = data[i]
                node_pos ^= c
                unit = units[node_pos]
                if (unit & 0x800000FF) != c:  # label mismatch
                    break
                node_pos ^= (unit >> 10) << ((unit & 0x200) >> 6)
                i += 1
                if unit & 0x100:  # has_leaf
                    best_len = i - pos
                    best_value = units[node_pos] & 0x7FFFFFFF
        if best_len == 0 or best_value >= len(self._normalized):
            first = data[pos]
            if first < 0x80:
                length = 1
            elif (first >> 5) == 0x6:
                length = 2
            elif (first >> 4) == 0xE:
                length = 3
            elif (first >> 3) == 0x1E:
                length = 4
            else:
                length = 1
            if length > 1:
                for k in range(1, length):
                    if pos + k >= n or (data[pos + k] & 0xC0) != 0x80:
                        length = 1
                        break
            return data[pos:pos + length], length
        return self._replacement(best_value), best_len

    def normalize(self, text):
        if not text:
            return ""
        raw = text.encode("utf-8")
        n = len(raw)
        pos = 0
        prefix = self.normalize_prefix
        ascii_table = self._ascii
        strip = self._remove_extra_whitespaces
        if strip:
            while pos < n:
                p, adv = prefix(raw, pos)
                if p != b" ":
                    break
                pos += adv
        if pos >= n:
            return ""
        out = bytearray()
        space = self._space_symbol
        if self._add_dummy_prefix:
            out += space
        is_prev_space = strip
        two_byte = self._two_byte
        last = n - 1
        while pos < n:
            b = raw[pos]
            if b < 0x80 and (pos == last or ((b << 8) | raw[pos + 1]) not in two_byte):
                p = ascii_table[b]
                adv = 1
            else:
                p, adv = prefix(raw, pos)
            pos += adv
            # remove heading spaces only if previous piece ended in whitespace
            if strip and is_prev_space:
                while p.startswith(b" "):
                    p = p[1:]
            if p:
                if 0x20 in p:
                    for byte in p:
                        if byte == 0x20:
                            out += space
                        else:
                            out.append(byte)
                    is_prev_space = p[-1] == 0x20
                else:
                    out += p
                    is_prev_space = False
        if strip:
            k = len(space)
            while out[-k:] == space:
                del out[-k:]
        return out.decode("utf-8")


class UsePureTokenizer:
    """Pure-python unigram tokenizer matching sentencepiece for the bundled model."""

    def __init__(self, model_path):
        with open(model_path, "rb") as f:
            data = f.read()
        top = _parse_fields(data, 0, len(data))

        # pieces (field 1, repeated) - legacy layout
        pieces = []
        scores = array("f")
        for st, ln in top[1]:
            p = _parse_fields(data, st, st + ln)
            if 1 in p:
                pieces.append(data[p[1][0][0]:p[1][0][0] + p[1][0][1]].decode("utf-8"))
                if 2 in p:
                    scores.append(struct.unpack("<f", data[p[2][0][0]:p[2][0][0] + 4])[0])
                else:
                    scores.append(0.0)
        self._pieces = pieces
        self._scores = scores  # array('f'): 0.5 MB instead of 4 MB of float objects
        self._unk_id = _UNK
        self._bos_id = _BOS
        self._eos_id = _EOS
        self._unk_score = min(self._scores) - 10.0
        self._max_len = max(len(p) for p in pieces)

        # flat prefix table: piece id + 1 for a real piece, 0 for a prefix that
        # only exists to keep the walk going (replaces a nested char trie).
        prefixes = {}
        for pid, piece in enumerate(pieces):
            for k in range(1, len(piece)):
                sub = piece[:k]
                if sub not in prefixes:
                    prefixes[sub] = 0
            prefixes[piece] = pid + 1
        self._prefixes = prefixes

        # charsmap from trainer_spec (field 3) -> its field 2
        charsmap = None
        if 3 in top:
            st3, ln3 = top[3][0]
            tf = _parse_fields(data, st3, st3 + ln3)
            if 2 in tf:
                charsmap = data[tf[2][0][0]:tf[2][0][0] + tf[2][0][1]]
        self._normalizer = _Normalizer(charsmap)

    @property
    def vocab_size(self):
        return len(self._pieces)

    def encode(self, texts, out_type=int, add_bos=True, add_eos=True):
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        results = []
        for text in batch:
            ids = self._encode_one(text)
            if out_type != int:
                pieces = self._pieces
                n = len(pieces)
                ids = [pieces[i] if i < n else "<unk>" for i in ids]
            if add_bos:
                ids = [self._bos_id if out_type == int else "<s>"] + ids
            if add_eos:
                ids = ids + [self._eos_id if out_type == int else "</s>"]
            results.append(ids)
        return results[0] if single else results

    def _encode_one(self, text):
        normalized = self._normalizer.normalize(text)
        if not normalized:
            return [self._unk_id]
        return self._viterbi(normalized)

    def _trie_prefixes(self, s, i):
        """Return [(piece_len, piece_id)] for pieces that are a prefix of s[i:]."""
        out = []
        get = self._prefixes.get
        stop = min(i + self._max_len, len(s))
        for j in range(i + 1, stop + 1):
            v = get(s[i:j])
            if v is None:
                break
            if v:
                out.append((j - i, v - 1))
        return out

    def _viterbi(self, normalized):
        """Exact port of sentencepiece unigram lattice Viterbi.

        For every position ``p``, all lattice nodes starting at ``p`` share the
        same best predecessor: the node ending at ``p`` with the highest
        backtrace score, earliest-starting on ties (the C++ strict-greater
        tie-breaking). That makes a plain left-to-right DP over positions
        equivalent to building the lattice: ``best[p]`` is the score of that
        predecessor, and the surviving node ending at ``p`` is (start, id).

        Nodes per start position are the vocabulary pieces in increasing length
        order, plus a single-character UNK node when no length-1 piece matches.
        """
        L = len(normalized)
        # every start position emits at least one node of length 1 (a piece or
        # the UNK fallback), so every best[p] gets a real score before it is read
        best = [0.0] + [-1e38] * L
        start_of = [0] * (L + 1)
        piece_of = [0] * (L + 1)
        get = self._prefixes.get
        scores = self._scores
        max_len = self._max_len
        unk_score = self._unk_score
        for i in range(L):
            base = best[i]
            stop = i + max_len
            if stop > L:
                stop = L
            has_single = False
            for j in range(i + 1, stop + 1):
                v = get(normalized[i:j])
                if v is None:
                    break
                if v:
                    score = base + scores[v - 1]
                    if score > best[j]:
                        best[j] = score
                        start_of[j] = i
                        piece_of[j] = v - 1
                    if j == i + 1:
                        has_single = True
            if not has_single:
                score = base + unk_score
                if score > best[i + 1]:
                    best[i + 1] = score
                    start_of[i + 1] = i
                    piece_of[i + 1] = self._unk_id

        ids = []
        p = L
        while p > 0:
            ids.append(piece_of[p])
            p = start_of[p]
        if not ids:
            return [self._unk_id]
        ids.reverse()
        return ids
