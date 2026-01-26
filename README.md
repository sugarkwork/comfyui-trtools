画像のタグ化のモデルと、アップスケーラーモデルをTensorRT を活用して NVIDIA GPU に最適化する事で、処理が凄く速くなります！

凄く速い！

WD14 Tagger (3.863s) -> TRT Tagger (0.240s)

<img width="1322" height="1064" alt="image" src="https://github.com/user-attachments/assets/b1edba60-68c2-440f-a493-08a88d0e0ed1" />


標準アップスケーラー（0.182s + 3.523s） -> TRT Upscaler (1.648s)

<img width="1111" height="1107" alt="image" src="https://github.com/user-attachments/assets/dea76fa5-43bb-4f43-9d9a-b78545cef131" />


注意点として、初回はモデルのダウンロード＋TensorRT への変換のため、数分フリーズしたかのように見えます。

モデルのダウンロードと変換は、初回しか行われませんので、処理が超高速になるのは２回目以降の実行です。
