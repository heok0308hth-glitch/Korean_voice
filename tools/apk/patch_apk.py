"""Rebuild the release APK with a new classes.dex, model manifest and version (unsigned output).
usage: patch_apk.py original.apk classes.dex model_manifest.json out_unsigned.apk old_version new_version new_code"""
import struct, sys, zipfile

src, dex, manifest, dst, old_v, new_v, new_code = sys.argv[1:8]
new_code = int(new_code)


def patch_axml(b: bytes) -> bytes:
    b = bytearray(b)
    # string pool chunk right after the 8-byte XML header
    sp = 8
    typ, hsz, csz, nstr, nsty, flags, sstart, _ = struct.unpack_from("<HHIIIIII", b, sp)
    assert typ == 0x0001
    utf8 = bool(flags & (1 << 8))
    offs = struct.unpack_from(f"<{nstr}I", b, sp + hsz)
    strings = []
    for o in offs:
        p = sp + sstart + o
        if utf8:
            n1 = b[p]; p += 2 if n1 & 0x80 else 1
            n = b[p]; p += 1
            if n & 0x80: n = ((n & 0x7F) << 8) | b[p]; p += 1
            strings.append((b[p:p + n].decode("utf-8"), p, n, "u8"))
        else:
            n = struct.unpack_from("<H", b, p)[0]; p += 2
            strings.append((b[p:p + 2 * n].decode("utf-16le"), p, n, "u16"))
    names = [s[0] for s in strings]
    for s, p, n, enc in strings:
        if s == old_v:
            assert len(new_v) == len(old_v)
            b[p:p + (2 * n if enc == "u16" else n)] = new_v.encode("utf-16le" if enc == "u16" else "utf-8")
    vc = names.index("versionCode")
    # walk chunks to the <manifest> start element and patch versionCode
    p = sp + csz
    while p < len(b):
        typ, hsz, csz = struct.unpack_from("<HHI", b, p)
        if typ == 0x0102:
            attr_start, attr_size, attr_count = struct.unpack_from("<HHH", b, p + 16 + 8)
            a0 = p + 16 + attr_start
            for i in range(attr_count):
                q = a0 + i * attr_size
                ns, name, raw, vsize, res0, dtype, data = struct.unpack_from("<IIIHBBI", b, q)
                if name == vc:
                    print("versionCode", data, "->", new_code)
                    struct.pack_into("<I", b, q + 16, new_code)
                    return bytes(b)
        p += csz
    raise SystemExit("versionCode not found")


zi = zipfile.ZipFile(src)
zo = zipfile.ZipFile(dst, "w")
for info in zi.infolist():
    n = info.filename
    if n.startswith("META-INF/") or n.startswith("assets/dexopt/"):
        continue
    data = zi.read(n)
    if n == "classes.dex":
        data = open(dex, "rb").read()
    elif n == "assets/model_manifest.json":
        data = open(manifest, "rb").read()
    elif n == "assets/THIRD_PARTY_NOTICES.txt":
        data = data.replace(b"https://github.com/heok0308hth-glitch/OpenVoice", b"https://github.com/heok0308hth-glitch/Korean_voice")
    elif n == "AndroidManifest.xml":
        data = patch_axml(data)
    zi_new = zipfile.ZipInfo(n, date_time=info.date_time)
    zi_new.compress_type = info.compress_type
    zi_new.external_attr = info.external_attr
    zo.writestr(zi_new, data)
zo.close()
print("wrote", dst)
