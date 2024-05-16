# linux-caffe
Using Caffe Model for AI Recognition on Linux Platform

## install dependence
### opencv

### ffmpeg 
  used for producting mp4 video which could play by html5's video tag
```
configure
./configure \
--enable-gpl \
--enable-libx264 \
--enable-libfdk-aac \
--enable-librtmp \
--enable-libopus \
--enable-libvpx \
--enable-libmp3lame \
--enable-libass \
--enable-libfreetype \
--enable-librtsp \
--prefix=/home/roota/workstation/opencv/ffmpeg \
--extra-cflags=-I/usr/local/include \
--extra-ldflags=-L/usr/local/lib
```