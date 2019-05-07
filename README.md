# Setup
動作環境
* Python 3.6.5
* Dependencies : numpy, matplotlib

# 課題
## ディレクトリ構成
```
    ├── output/              # 出力用ディレクトリ
    ├── src/
    |    ├── task1           # 課題1のコード
    |    ├── task2           # 課題2のコード
    |    ├── task3           # 課題3のコード
    |    ├── task4           # 課題4のコード
    |    ├── datasets        # データセット
    |    └── functions       # activation, lossなどの関数置き場
    ├── report.pdf           # レポート
    └── README.md
```

ディレクトリ構成は、`chainer-chemistry` を少し参考にし、  
functionsというディレクトリに再利用性の高いコードを置きました。

## 課題1
UnitTestを `task1/gnn.test.py` に書きました。  
以下のコマンドで実行できます。  
(**以下、全てのコマンドはrootから実行しないとエラーが出ることに注意して下さい。**)

```
$ python src/task1/gnn.test.py -v
```

## 課題2
勾配法による損失関数を最小化するコードを書きました。
以下のコマンドで実行できます。

```
$ python src/task2/gnn.py
```

実行後、最終的な損失の値が表示され、`task2/loss.png` に損失の値がプロットされたグラフが保存されていることが確認できると思います。(具体的なグラフの形は、`output/loss.png` と同じ形になると思います。)

## 課題3

課題3については、

* 上手く学習ができていない
* 処理にかなり時間がかかる

以上のことから、実行しない方が良いと思いますが、一応コードの実行方法は以下になります。  
また上手く学習できなかったことから、[デバックについてレポート (report.pdf)](./report.pdf)を記述したので、
そちらを参照して頂ければと思います。

SGDによる学習は、以下のコマンドで実行できます。  
実行結果は、`output/` 以下にあります。

```
$ python src/task3/sgd.py
```

Momentum-SGDによる学習は、以下のコマンドで実行できます。

```
$ python src/task3/msgd.py
```

## 課題4
課題3の時点で学習がうまくいかなかったので、  
Adamの実装だけ `src/task4/updater.py` においてあるのでご確認して頂ければと思います。
