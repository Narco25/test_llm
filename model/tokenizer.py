class SimpleTokenizer:
    """A deterministic character-level tokenizer with four reserved tokens."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self) -> None:
        alphabet = "\n\t" + "".join(chr(code) for code in range(32, 127))
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]
        self.token_to_id = {
            token: index for index, token in enumerate(special_tokens)
        }
        self.token_to_id.update(
            {character: index + len(special_tokens) for index, character in enumerate(alphabet)}
        )
        self.id_to_token = {
            index: token for token, index in self.token_to_id.items()
        }

        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        self.unk_id = self.token_to_id[self.UNK_TOKEN]
        self.bos_id = self.token_to_id[self.BOS_TOKEN]
        self.eos_id = self.token_to_id[self.EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(character, self.unk_id) for character in text]

    def decode(self, tokens: list[int]) -> str:
        characters = []
        for token in tokens:
            if token in (self.pad_id, self.bos_id, self.eos_id):
                continue
            characters.append(self.id_to_token.get(token, self.UNK_TOKEN))
        return "".join(characters)