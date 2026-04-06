#!/usr/bin/env python3
"""Export tokenizer.model (SentencePiece) to tokenizer.bin (flat binary).

Format:
    [max_token_length: u32]
    for each token (0 .. vocab_size-1):
        [score: f32] [len: u32] [bytes: u8 * len]

Usage:
    pip install sentencepiece
    python tools/host/export_tokenizer.py --tokenizer-model weights/tokenizer.model
"""

import argparse, struct, os


def export(tokenizer_model, output_path):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=tokenizer_model)
    vocab_size = sp.vocab_size()

    tokens = []
    scores = []
    for i in range(vocab_size):
        t = sp.id_to_piece(i)
        s = sp.get_score(i)
        if i == sp.bos_id():
            t = '\n<s>\n'
        elif i == sp.eos_id():
            t = '\n</s>\n'
        t = t.replace('▁', ' ')
        b = t.encode('utf-8')
        tokens.append(b)
        scores.append(s)

    max_len = max(len(t) for t in tokens)

    with open(output_path, 'wb') as f:
        f.write(struct.pack('I', max_len))
        for b, s in zip(tokens, scores):
            f.write(struct.pack('fI', s, len(b)))
            f.write(b)

    size = os.path.getsize(output_path)
    print(f"exported {vocab_size} tokens to {output_path} ({size} bytes)")
    print(f"  max_token_length: {max_len}")


def test_encode(tokenizer_model):
    """Print reference encodings for Pi-side verification."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=tokenizer_model)

    prompts = [
        "Once upon a time",
        "A cat sat on",
        "Hello",
    ]
    for text in prompts:
        ids = sp.encode(text)
        pieces = [sp.id_to_piece(i).replace('▁', ' ') for i in ids]
        print(f'encode("{text}")')
        print(f'  with BOS: [1, {", ".join(str(i) for i in ids)}]')
        for tok_id, piece in zip(ids, pieces):
            print(f'    {tok_id:5d} -> "{piece}"')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-model', required=True,
                        help='path to tokenizer.model (SentencePiece)')
    parser.add_argument('--output', default=None,
                        help='output path (default: weights/tokenizer.bin)')
    parser.add_argument('--test', action='store_true',
                        help='print reference encodings instead of exporting')
    args = parser.parse_args()

    if args.test:
        test_encode(args.tokenizer_model)
    else:
        output = args.output or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'weights', 'tokenizer.bin')
        export(args.tokenizer_model, output)
