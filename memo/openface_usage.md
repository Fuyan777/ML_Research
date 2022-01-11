1.ホスト側のマウントして、algebr/openface:latest を実行
```
docker run -v /Volumes/mac-ssd/movie:/home/openface-build/mount/ -it --rm algebr/openface:latest
```
※$PWDは指定する必要がある

（proceedをホスト側へコピー）※1でローカルをmountしてるので、新たに動画を追加するときにだけ利用
```
docker cp ~/Documents/face-movie e8b6a2899b58:/home/openface-build/mount/
```

2.動画解析
```
build/bin/FeatureExtraction -f /home/openface-build/mount/Documents/face-movie/
```

3.docker側からホストにmount
```
docker cp e8b6a2899b58:/home/openface-build/processed/~.csv ~/Documents/
```


mdプレビュー
command + K -> V