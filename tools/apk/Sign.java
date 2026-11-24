import com.android.apksig.ApkSigner;
import com.android.apksig.ApkVerifier;

import java.io.File;
import java.io.FileInputStream;
import java.security.KeyStore;
import java.security.PrivateKey;
import java.security.cert.X509Certificate;
import java.util.List;

/** Sign (v2 + v3, aligned) and verify an APK with a PKCS12 key. usage: Sign in.apk out.apk keystore.p12 alias password */
public class Sign {
    public static void main(String[] a) throws Exception {
        char[] pw = a[4].toCharArray();
        KeyStore ks = KeyStore.getInstance("PKCS12");
        try (FileInputStream in = new FileInputStream(a[2])) { ks.load(in, pw); }
        PrivateKey key = (PrivateKey) ks.getKey(a[3], pw);
        X509Certificate cert = (X509Certificate) ks.getCertificate(a[3]);
        ApkSigner.SignerConfig sc = new ApkSigner.SignerConfig.Builder("KoreanVoice", key, List.of(cert)).build();
        new ApkSigner.Builder(List.of(sc)).setInputApk(new File(a[0])).setOutputApk(new File(a[1]))
                .setMinSdkVersion(26).setV1SigningEnabled(false).setV2SigningEnabled(true).setV3SigningEnabled(true)
                .build().sign();
        ApkVerifier.Result r = new ApkVerifier.Builder(new File(a[1])).build().verify();
        System.out.println("verified=" + r.isVerified() + " v2=" + r.isVerifiedUsingV2Scheme() + " v3=" + r.isVerifiedUsingV3Scheme()
                + " errors=" + r.getErrors());
        java.security.MessageDigest md = java.security.MessageDigest.getInstance("SHA-256");
        StringBuilder sb = new StringBuilder();
        for (byte x : md.digest(cert.getEncoded())) sb.append(String.format("%02x", x));
        System.out.println("cert sha256=" + sb);
    }
}
