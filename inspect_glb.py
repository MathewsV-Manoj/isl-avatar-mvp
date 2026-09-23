import json
import struct
import sys

path = sys.argv[1]
data = open(path, "rb").read()
magic, version, total = struct.unpack_from("<III", data, 0)
offset = 12
document = None
chunks = []
while offset < total:
    size, kind = struct.unpack_from("<II", data, offset)
    offset += 8
    payload = data[offset : offset + size]
    offset += size
    chunks.append((hex(kind), size))
    if kind == 0x4E4F534A:
        document = json.loads(payload.decode("utf-8"))

print({
    "magic": hex(magic),
    "version": version,
    "chunks": chunks,
    "nodes": len(document.get("nodes", [])),
    "skins": len(document.get("skins", [])),
    "meshes": len(document.get("meshes", [])),
    "animations": len(document.get("animations", [])),
})
for skin in document.get("skins", []):
    joints = [document["nodes"][index].get("name") for index in skin.get("joints", [])]
    print("skin", skin.get("name"), "joints", len(joints))
    print(joints)
for mesh in document.get("meshes", []):
    target_count = sum(len(primitive.get("targets", [])) for primitive in mesh.get("primitives", []))
    if target_count:
        print("morph mesh", mesh.get("name"), "target accessors", target_count)
