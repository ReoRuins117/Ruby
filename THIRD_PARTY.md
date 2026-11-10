# Third-party notices

## nnue_data_compress (binpack format)

`convert_binpack.py` reads the Stockfish `.binpack` format. Its decoder follows the format from
[Sopel97/nnue_data_compress](https://github.com/Sopel97/nnue_data_compress), used under the MIT License:

```
Copyright 2019 Tomasz Sobczyk

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Training data

If networks are trained on Stockfish binpack data, that data comes from
[official-stockfish/master-binpacks](https://huggingface.co/datasets/official-stockfish/master-binpacks)
(Stockfish and Leela Chess Zero projects), made available under the
[Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/).

## Fathom (Syzygy tablebase probing)

`engine/fathom/` is [Fathom](https://github.com/jdart1/Fathom) (commit c9c6fef), which Ruby uses to
read Syzygy endgame tablebases. It is used under the MIT License; the full text is in
`engine/fathom/LICENSE`:

```
Copyright (c) 2013-2018 Ronald de Man
Copyright (c) 2015 basil00
Copyright (c) 2016-2025 by Jon Dart
```

The tablebase files themselves are not part of this repository. Ruby's lichess-bot setup uses the
3-4-5 piece tables from tablebase.lichess.ovh.
