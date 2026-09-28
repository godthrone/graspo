"""§12.4 文件头职责声明：对给定目录逐个文件算 sha256，汇总「文件数 + 总指纹」用于对照实验机。

边界：只读、只算指纹、只打印；不写任何文件、不改目录内容、不做判定。
"""
import hashlib,pathlib,sys
d=pathlib.Path(sys.argv[1])
entries=[]
for p in sorted(d.iterdir()):
    if p.is_file(): entries.append((p.name,hashlib.sha256(p.read_bytes()).hexdigest()))
h=hashlib.sha256()
for n,s in sorted(entries): h.update(f"{s}  {n}\n".encode())
print("文件数",len(entries),"总指纹",h.hexdigest())
print("期望    54 总指纹 7a230bad535a79cbd9718c7169a1fbabac156948b5bb991a762ee3bcca6f0e47")
