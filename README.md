# Setup
動作環境
* Python 3.6.5
* Dependencies : numpy, matplotlib

# 課題
## ディレクトリ構成
```
    ├── report/              # report用ディレクトリ
    ├── src/
    |    ├── task1           # 課題1のコード
    |    ├── task2           # 課題2のコード
    |    ├── task3           # 課題3のコード
    |    ├── task4           # 課題4のコード
    |    ├── datasets        # データセット
    |    └── functions       # activation, lossなどの関数置き場
    └── README.md
```

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

実行後、最終的な損失の値が表示され、`task2/loss.png` に損失の値がプロットされたグラフが保存されていることが確認できると思います。

## 課題3
SGDによる学習の様子は、以下のコマンドで実行できます。

```
$ python src/task3/sgd.py
```

Momentum-SGDによる学習の様子は、以下のコマンドで実行できます。

```
$ python src/task3/msgd.py
```

## 課題4
課題3の時点で学習がうまくいかなかったので、  
Adamの実装だけ `src/task4/updater.py` においてあるのでご確認して頂ければと思います。
