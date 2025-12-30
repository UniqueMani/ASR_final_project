def longest_common_prefix(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]

class SubtitleStabilizer:
    def __init__(self, stability_rounds: int = 2):
        self.stability_rounds = max(2, stability_rounds)
        self._history = []
        self.committed = ""
        self.partial = ""

    def update(self, new_partial: str):
        new_partial = (new_partial or "").strip()
        self._history.append(new_partial)
        if len(self._history) > self.stability_rounds:
            self._history.pop(0)

        prefix = self._history[0]
        for s in self._history[1:]:
            prefix = longest_common_prefix(prefix, s)
            if not prefix:
                break

        if len(prefix) > len(self.committed):
            self.committed = prefix

        if new_partial.startswith(self.committed):
            self.partial = new_partial[len(self.committed):].lstrip()
        else:
            self.partial = new_partial

        return self.committed, self.partial
