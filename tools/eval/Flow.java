import io.github.koreanvoice.audio.Mp3Encoder;
import io.github.koreanvoice.expressive.*;

import java.io.*;
import java.lang.reflect.Constructor;
import java.util.List;

/** The app's real flow on the desktop: learn a voice from files (VoiceBank), auto-learn briefly (AutoTuner),
 *  then render a marked-up script (ExpressiveJob) to MP3. */
public class Flow {
    public static void main(String[] a) throws Exception {
        File dir = new File(a[0]);
        File bankDir = new File(a[1]);
        File out = new File(a[2]);
        String script = new String(java.nio.file.Files.readAllBytes(new File(a[3]).toPath()), "UTF-8");
        int tunerBudget = Integer.parseInt(a[4]);
        CbEngine eng = new CbEngine(dir, 4, "s3gen_turbo.onnx");
        VoiceBank bank = new VoiceBank(bankDir);
        long t0 = System.nanoTime();
        for (int i = 5; i < a.length; i++) {
            int[] sr = new int[1];
            float[] pcm = Synth.read(new File(a[i]), sr);
            VoiceBank.IngestReport rep = bank.ingest(eng, pcm, sr[0], new File(a[i]).getName(),
                    (O1.e) (f, m) -> null, (O1.a) () -> Boolean.FALSE);
            System.out.println("INGEST " + a[i] + " -> " + rep);
        }
        System.out.printf("bank: %d segments, %.1fs, learn time %.1fs%n", bank.getSegments().size(), bank.getTotalSeconds(), (System.nanoTime() - t0) / 1e9);
        eng.releaseRegistrationModels();

        if (tunerBudget > 0) {
            Constructor<AutoTuner.Config> c = AutoTuner.Config.class.getDeclaredConstructor(int.class, int.class, int.class, List.class, List.class,
                    long.class, double.class, double.class, double.class, int.class, Class.forName("P1.f"));
            c.setAccessible(true);
            AutoTuner.Config cfg = c.newInstance(1, tunerBudget, 2, null, null, 0L, 0.0, 0.0, 0.0, 504, null);
            long t1 = System.nanoTime();
            AutoTuner.Result res = new AutoTuner(eng, bank, cfg).run((O1.c) p -> null, (O1.a) () -> Boolean.FALSE);
            System.out.printf("AUTOTUNE baseline=%.4f best=%.4f accepted=%b evaluated=%d time=%.0fs heldOut=%s%n", res.getBaseline(),
                    res.getBest() == null ? Double.NaN : res.getBest().getScore(), res.getAccepted(), res.getEvaluated(), (System.nanoTime() - t1) / 1e9, res.getHeldOut());
            bank.applyAutoLearning(res);
        }

        long t2 = System.nanoTime();
        ExpressiveJob job = new ExpressiveJob(eng, bank, new File(out.getParentFile(), "jobcache"), script, new ExpressiveJob.Options());
        job.render((O1.a) () -> Boolean.FALSE, (O1.c) p -> null);
        try (OutputStream os = new BufferedOutputStream(new FileOutputStream(out))) {
            job.assembleMp3(os, 128, new Mp3Encoder.Id3Info("테스트", bank.getName(), "AI 생성 음성"));
        }
        job.assembleWav(new File(out.getPath().replace(".mp3", ".wav")));
        System.out.printf("RENDER chunks=%d mp3=%d bytes time=%.0fs%n", job.completedChunks(), out.length(), (System.nanoTime() - t2) / 1e9);
        eng.close();
    }
}
