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
charsmap in field 2). Pieces carry only their surface string (no scores), so
every piece is scored 0 and Viterbi is decided by lattice insertion order.
"""

import struct
from collections import defaultdict

_UNK = 0
_BOS = 1
_EOS = 2

_MAX_TRIE_RESULTS = 64


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


class _DoubleArrayTrie:
    """Darts-clone double-array common-prefix search (unit size 4)."""

    def __init__(self, trie_blob):
        n_units = len(trie_blob) // 4
        self.units = struct.unpack(f"<{n_units}I", trie_blob)

    @staticmethod
    def _has_leaf(unit):
        return ((unit >> 8) & 1) == 1

    @staticmethod
    def _value(unit):
        return unit & ((1 << 31) - 1)

    @staticmethod
    def _label(unit):
        return unit & ((1 << 31) | 0xFF)

    @staticmethod
    def _offset(unit):
        return (unit >> 10) << ((unit & (1 << 9)) >> 6)

    def common_prefix_search(self, key):
        """Return [(value, byte_length)] for keys that prefix `key`."""
        unit = self.units[0]
        node_pos = 0 ^ self._offset(unit)
        results = []
        for i in range(len(key)):
            node_pos ^= key[i]
            unit = self.units[node_pos]
            if self._label(unit) != key[i]:
                break
            node_pos ^= self._offset(unit)
            if self._has_leaf(unit):
                results.append((self._value(self.units[node_pos]), i + 1))
        return results


class _Normalizer:
    def __init__(self, charsmap, add_dummy_prefix=True,
                 remove_extra_whitespaces=True, escape_whitespaces=True):
        if charsmap:
            trie_size = struct.unpack("<I", charsmap[:4])[0]
            trie = charsmap[4:4 + trie_size]
            normalized = charsmap[4 + trie_size:]
            self._trie = _DoubleArrayTrie(trie)
            self._normalized = normalized
        else:
            self._trie = None
            self._normalized = None
        self._add_dummy_prefix = add_dummy_prefix
        self._remove_extra_whitespaces = remove_extra_whitespaces
        self._space_symbol = b"\xe2\x96\x81" if escape_whitespaces else b" "

    def normalize_prefix(self, input_bytes):
        """Return (normalized_prefix_bytes, consumed_byte_len)."""
        if not input_bytes:
            return b"", 0
        longest_length = 0
        longest_value = 0
        if self._trie is not None:
            for value, length in self._trie.common_prefix_search(input_bytes):
                if length > longest_length:
                    longest_length = length
                    longest_value = value
        if longest_length == 0 or longest_length > len(input_bytes) or \
                longest_value >= len(self._normalized):
            first = input_bytes[0]
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
                    if k >= len(input_bytes) or (input_bytes[k] & 0xC0) != 0x80:
                        length = 1
                        break
            return input_bytes[:length], length
        end = self._normalized.index(b"\x00", longest_value)
        return self._normalized[longest_value:end], longest_length

    def normalize(self, text):
        if not text:
            return ""
        raw = text.encode("utf-8")
        if self._remove_extra_whitespaces:
            while raw:
                p, adv = self.normalize_prefix(raw)
                if p != b" ":
                    break
                raw = raw[adv:]
        if not raw:
            return ""
        out = bytearray()
        if self._add_dummy_prefix:
            out += self._space_symbol
        is_prev_space = self._remove_extra_whitespaces
        while raw:
            p, adv = self.normalize_prefix(raw)
            sp = p
            # remove heading spaces only if previous piece ended in whitespace
            if self._remove_extra_whitespaces and is_prev_space:
                while sp.startswith(b" "):
                    sp = sp[1:]
            if sp:
                for byte in sp:
                    if byte == 0x20:
                        out += self._space_symbol
                    else:
                        out.append(byte)
                is_prev_space = sp.endswith(b" ")
            raw = raw[adv:]
        if self._remove_extra_whitespaces:
            while out.endswith(self._space_symbol):
                del out[-len(self._space_symbol):]
        return out.decode("utf-8")


class UsePureTokenizer:
    """Pure-python unigram tokenizer matching sentencepiece for the bundled model."""

    def __init__(self, model_path):
        with open(model_path, "rb") as f:
            self._data = f.read()
        top = _parse_fields(self._data, 0, len(self._data))

        # pieces (field 1, repeated) - legacy layout
        self._pieces = []
        self._scores = []
        for st, ln in top[1]:
            p = _parse_fields(self._data, st, st + ln)
            if 1 in p:
                self._pieces.append(
                    self._data[p[1][0][0]:p[1][0][0] + p[1][0][1]].decode("utf-8"))
                if 2 in p:
                    self._scores.append(
                        struct.unpack("<f", self._data[p[2][0][0]:p[2][0][0] + 4])[0])
                else:
                    self._scores.append(0.0)
        self._id = {piece: i for i, piece in enumerate(self._pieces)}
        self._unk_id = 0
        self._bos_id = 1
        self._eos_id = 2
        self._min_score = min(self._scores)
        # prefix trie for fast piece lookup (by codepoint)
        self._trie = {}
        for pid, piece in enumerate(self._pieces):
            node = self._trie
            for ch in piece:
                node = node.setdefault(ch, {})
            node[None] = pid

        # charsmap from trainer_spec (field 3) -> its field 2
        charsmap = None
        if 3 in top:
            st3, ln3 = top[3][0]
            tf = _parse_fields(self._data, st3, st3 + ln3)
            if 2 in tf:
                charsmap = self._data[tf[2][0][0]:tf[2][0][0] + tf[2][0][1]]
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
            if out_type == int:
                ids = ids
            else:
                ids = [self._pieces[i] if i < len(self._pieces) else "<unk>"
                       for i in ids]
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
        node = self._trie
        out = []
        j = i
        n = len(s)
        while j < n and s[j] in node:
            node = node[s[j]]
            j += 1
            if None in node:
                out.append((j - i, node[None]))
        return out

    def _viterbi(self, normalized):
        """Exact port of sentencepiece unigram lattice Viterbi.

        node order per start position = trie prefix order (increasing length),
        then a single-char UNK fallback node. For every position `pos`, all
        nodes starting at `pos` pick the same best predecessor (the node ending
        at `pos` with the highest backtrace score, first-on-tie). This mirrors
        the C++ strict-greater tie-breaking.
        """
        L = len(normalized)
        unk_score = self._min_score - 10.0
        # nodes ending at each position, in insertion order: tuples (start, length, id, score)
        end = [[] for _ in range(L + 1)]
        end[0].append((0, 0, -1, 0.0))  # BOS
        # per-start insertion order preserved when building end lists
        for i in range(L):
            has_single = False
            for ln, pid in self._trie_prefixes(normalized, i):
                node = (i, ln, pid, self._scores[pid])
                end[i + ln].append(node)
                if ln == 1:
                    has_single = True
            if not has_single:
                end[i + 1].append((i, 1, self._unk_id, unk_score))

        # bts[node] = best backtrace score; prev[node] = best predecessor
        bts = {}
        prev = {}
        bts[(0, 0, -1, 0.0)] = 0.0
        # process positions left to right; bts of a node ending at pos depends on
        # best_lnode at pos, which depends on bts of nodes ending at pos (already computed
        # for earlier starts). We must compute in order of end position.
        best_lnode_at = [None] * (L + 1)  # best predecessor for nodes ending... actually for rnode at start=end
        # We need, for each position p, the best lnode in end[p] (nodes ending at p).
        # Those lnodes' bts are known once all nodes ending at p have been assigned bts,
        # which happens when their start's best predecessor is known. Process p in order.
        for p in range(L + 1):
            # compute bts for all nodes in end[p] first? They need best_lnode_at[start].
            # A node (s,ln,id,sc) in end[p] has start s = p-ln. Its bts = best_lnode_at[s].bts + sc.
            for node in end[p]:
                s = node[0]
                if s == 0 and node[2] == -1:
                    continue  # BOS already set
                bl = best_lnode_at[s]
                if bl is None:
                    prev[node] = None
                    bts[node] = 0.0
                else:
                    prev[node] = bl
                    bts[node] = bts[bl] + node[3]
            # now best_lnode_at[p] = argmax over end[p] of bts (first on tie)
            best = None
            for node in end[p]:
                if best is None or bts[node] > bts[best]:
                    best = node
            best_lnode_at[p] = best

        # EOS: a node at L; its best predecessor is best_lnode_at[L]
        eos_prev = best_lnode_at[L]
        ids = []
        cur = eos_prev
        while cur is not None and cur[2] != -1:
            ids.append(cur[2])
            cur = prev.get(cur)
        ids.reverse()
        if not ids:
            return [self._unk_id]
        return ids
