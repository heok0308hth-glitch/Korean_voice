import io.github.koreanvoice.expressive.CbEngine;
import io.github.koreanvoice.expressive.CbParams;
import io.github.koreanvoice.expressive.CbVoice;

import javax.sound.sampled.*;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Runs the APK's own CbEngine (dex2jar) on the desktop: clone a voice from a wav and read Korean text. */
public class Synth {
    static float[] read(File f, int[] sr) throws Exception {
        AudioInputStream in = AudioSystem.getAudioInputStream(f);
        AudioFormat fmt = in.getFormat();
        byte[] b = in.readAllBytes();
        int ch = fmt.getChannels();
        int n = b.length / 2 / ch;
        float[] x = new float[n];
        ByteBuffer bb = ByteBuffer.wrap(b).order(fmt.isBigEndian() ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) { float s = 0; for (int c = 0; c < ch; c++) s += bb.getShort() / 32768f; x[i] = s / ch; }
        sr[0] = (int) fmt.getSampleRate();
        return x;
    }

    static void write(File f, float[] x, int sr) throws Exception {
        byte[] b = new byte[x.length * 2];
        for (int i = 0; i < x.length; i++) {
            int v = Math.max(-32768, Math.min(32767, Math.round(x[i] * 32767)));
            b[2 * i] = (byte) v; b[2 * i + 1] = (byte) (v >> 8);
        }
        AudioFormat fmt = new AudioFormat(sr, 16, 1, true, false);
        AudioSystem.write(new AudioInputStream(new ByteArrayInputStream(b), fmt, x.length), AudioFileFormat.Type.WAVE, f);
    }

    public static void main(String[] a) throws Exception {
        File dir = new File(a[0]);
        String decoder = a[1];
        File ref = new File(a[2]);
        File outDir = new File(a[3]); outDir.mkdirs();
        int[] sr = new int[1];
        float[] pcm = read(ref, sr);
        long t0 = System.nanoTime();
        CbEngine eng = new CbEngine(dir, 4, decoder);
        CbVoice v = eng.buildVoice(pcm, sr[0], null);
        System.out.printf("voice built in %.1fs: genTokens=%d t3Tokens=%d melFrames=%d%n", (System.nanoTime() - t0) / 1e9,
                v.getGenTokens().length, v.getT3Tokens().length, v.getGenMelFrames());
        for (int i = 4; i < a.length; i++) {
            long t1 = System.nanoTime();
            CbEngine.Result r = eng.synthesize(a[i], v, new CbParams(), (O1.a) () -> Boolean.FALSE);
            double el = (System.nanoTime() - t1) / 1e9;
            File f = new File(outDir, String.format("s%02d.wav", i - 4));
            write(f, r.getAudio(), 24000);
            System.out.printf("RESULT %s tokens=%d forced=%b audio=%.2fs time=%.1fs text=%s%n", f.getName(), r.getTokens(), r.getForcedStop(),
                    r.getAudio().length / 24000.0, el, a[i]);
        }
        eng.close();
    }
}
