@echo off

set ComplilerFlags=/EHsc -MTd -nologo -Gm- -GR- -EHa- -Od -Oi -WX -W4 -wd4201 -wd4100 -wd4189 -wd4505 -wd4127 -DNN_INTERNAL=1 -FC -Z7
set LinkerFlags=-incremental:no -opt:ref

pushd w:\nn\build
REM main debug build

cl %ComplilerFlags% w:\nn\code\nn.cpp -Fmnn.map /link %LinkerFlags%
popd