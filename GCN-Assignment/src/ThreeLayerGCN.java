import java.io.*;
import java.nio.file.*;
import java.util.*;

public class ThreeLayerGCN {

    static final String CONTENT = "dataset/citeseer/citeseer.content";
    static final String CITES = "dataset/citeseer/citeseer.cites";
    static final String OUT = "results/citeseer_three_layer";

    static final int H1 = 32;
    static final int H2 = 16;
    static final int EPOCHS = 200;

    static final double LR = 0.01;
    static final double L2 = 0.0005;
    static final long SEED = 42;

    static class Data {
        int n, f, c;
        String[] ids, names;
        double[][] x;
        int[] y;
        List<Integer>[] nbr;
        List<Double>[] wt;
        boolean[] train, val, test;
    }

    static class Model {
        double[][] w0, w1, w2;
        double[][] b0, b1, b2;

        Model(int in, int h1, int h2, int out, Random r) {
            w0 = init(in, h1, r, Math.sqrt(2.0 / in));
            w1 = init(h1, h2, r, Math.sqrt(2.0 / h1));
            w2 = init(h2, out, r, Math.sqrt(2.0 / h2));

            b0 = new double[1][h1];
            b1 = new double[1][h2];
            b2 = new double[1][out];
        }
    }

    static class Fwd {
        double[][] ax, u0, h0, ah0;
        double[][] u1, h1, ah1, z, p;
    }

    public static void main(String[] args) throws Exception {

        Data d = load();
        Files.createDirectories(Paths.get(OUT));

        System.out.println("===== CITESEER: THREE-LAYER GCN =====");
        System.out.println("Nodes: " + d.n);
        System.out.println("Features per node: " + d.f);
        System.out.println("Classes: " + d.c);
        System.out.println("Undirected citation edges: " + edgeCount(d) / 2);
        System.out.println("Training nodes: " + count(d.train));
        System.out.println("Validation nodes: " + count(d.val));
        System.out.println("Testing nodes: " + count(d.test));

        Model model = new Model(
                d.f, H1, H2, d.c, new Random(SEED));

        Model best = null;
        double bestF1 = -1;

        try (PrintWriter log = new PrintWriter(
                new FileWriter(OUT + "/training.csv"))) {

            log.println("epoch,train_loss,val_accuracy,val_macro_f1");

            for (int epoch = 1; epoch <= EPOCHS; epoch++) {

                Fwd f = forward(d, model);
                double trainLoss = loss(d, f.p, d.train);

                backward(d, model, f);

                Fwd after = forward(d, model);

                double valAccuracy = accuracy(d, after.p, d.val);
                double valF1 = macroF1(d, after.p, d.val);

                log.printf(Locale.US, "%d,%.8f,%.6f,%.6f%n",
                        epoch, trainLoss, valAccuracy, valF1);

                if (valF1 > bestF1) {
                    bestF1 = valF1;
                    best = copy(model);
                }

                if (epoch == 1 || epoch % 20 == 0) {
                    System.out.printf(Locale.US,
                            "Epoch %3d | Loss %.4f | Validation Accuracy %.4f | Validation Macro-F1 %.4f%n",
                            epoch, trainLoss, valAccuracy, valF1);
                }
            }
        }

        if (best != null) {
            model = best;
        }

        Fwd finalForward = forward(d, model);

        double acc = accuracy(d, finalForward.p, d.test);
        double precision = macroPrecision(d, finalForward.p, d.test);
        double recall = macroRecall(d, finalForward.p, d.test);
        double f1 = macroF1(d, finalForward.p, d.test);

        System.out.println("\n===== TEST RESULTS =====");

        System.out.printf(Locale.US, "Accuracy        : %.4f%n", acc);
        System.out.printf(Locale.US, "Macro Precision : %.4f%n", precision);
        System.out.printf(Locale.US, "Macro Recall    : %.4f%n", recall);
        System.out.printf(Locale.US, "Macro F1-score  : %.4f%n", f1);

        printClassMetrics(d, finalForward.p);
        printConfusionMatrix(d, finalForward.p);

        try (PrintWriter out = new PrintWriter(
                new FileWriter(OUT + "/metrics.txt"))) {

            out.println("Dataset: CiteSeer");
            out.println("Model: Three-layer GCN");
            out.println("Nodes: " + d.n);
            out.println("Features per node: " + d.f);
            out.println("Classes: " + d.c);
            out.println("Training nodes: " + count(d.train));
            out.println("Validation nodes: " + count(d.val));
            out.println("Testing nodes: " + count(d.test));

            out.printf(Locale.US, "Accuracy: %.6f%n", acc);
            out.printf(Locale.US, "Macro Precision: %.6f%n", precision);
            out.printf(Locale.US, "Macro Recall: %.6f%n", recall);
            out.printf(Locale.US, "Macro F1-score: %.6f%n", f1);
        }

        savePredictions(d, finalForward.p);

        System.out.println("\nResults saved in: " + OUT);
    }

    // Load CiteSeer paper features, labels and citation edges
    static Data load() throws IOException {

        if (!Files.exists(Paths.get(CONTENT))
                || !Files.exists(Paths.get(CITES))) {
            throw new FileNotFoundException(
                    "CiteSeer files not found.\nExpected:\n"
                    + CONTENT + "\n" + CITES);
        }

        List<String[]> rows = new ArrayList<>();
        TreeSet<String> labels = new TreeSet<>();
        Map<String, Integer> index = new LinkedHashMap<>();

        int featureCount = -1;

        for (String line : Files.readAllLines(Paths.get(CONTENT))) {

            line = line.trim();

            if (line.isEmpty()) {
                continue;
            }

            String[] p = line.split("\\s+");

            if (p.length < 3) {
                continue;
            }

            if (featureCount == -1) {
                featureCount = p.length - 2;
            }

            if (p.length - 2 != featureCount) {
                throw new IOException(
                        "Inconsistent feature count near paper " + p[0]);
            }

            if (index.containsKey(p[0])) {
                throw new IOException("Duplicate paper ID: " + p[0]);
            }

            index.put(p[0], rows.size());
            rows.add(p);
            labels.add(p[p.length - 1]);
        }

        if (rows.isEmpty()) {
            throw new IOException("No paper records found.");
        }

        Data d = new Data();

        d.n = rows.size();
        d.f = featureCount;
        d.c = labels.size();

        d.ids = new String[d.n];
        d.x = new double[d.n][d.f];
        d.y = new int[d.n];
        d.names = labels.toArray(new String[0]);

        Map<String, Integer> labelIndex = new HashMap<>();

        for (int i = 0; i < d.c; i++) {
            labelIndex.put(d.names[i], i);
        }

        for (int i = 0; i < d.n; i++) {

            String[] p = rows.get(i);
            d.ids[i] = p[0];

            for (int j = 0; j < d.f; j++) {
                d.x[i][j] = Double.parseDouble(p[j + 1]);
            }

            d.y[i] = labelIndex.get(p[p.length - 1]);
        }

        @SuppressWarnings("unchecked")
        Set<Integer>[] adjacency = new Set[d.n];

        for (int i = 0; i < d.n; i++) {
            adjacency[i] = new TreeSet<>();
        }

        int unknown = 0;

        for (String line : Files.readAllLines(Paths.get(CITES))) {

            line = line.trim();

            if (line.isEmpty()) {
                continue;
            }

            String[] p = line.split("\\s+");

            if (p.length < 2) {
                continue;
            }

            Integer u = index.get(p[0]);
            Integer v = index.get(p[1]);

            if (u == null || v == null) {
                unknown++;
                continue;
            }

            if (!u.equals(v)) {
                adjacency[u].add(v);
                adjacency[v].add(u);
            }
        }

        d.nbr = new List[d.n];
        d.wt = new List[d.n];

        double[] degree = new double[d.n];

        // Add one self-loop to every node
        for (int i = 0; i < d.n; i++) {
            degree[i] = adjacency[i].size() + 1.0;
        }

        // Symmetrically normalized adjacency:
        // A_hat = D^(-1/2) (A + I) D^(-1/2)
        for (int i = 0; i < d.n; i++) {

            d.nbr[i] = new ArrayList<>();
            d.wt[i] = new ArrayList<>();

            d.nbr[i].add(i);
            d.wt[i].add(1.0 / degree[i]);

            for (int j : adjacency[i]) {
                d.nbr[i].add(j);
                d.wt[i].add(
                        1.0 / Math.sqrt(degree[i] * degree[j]));
            }
        }

        if (unknown > 0) {
            System.out.println(
                    "Warning: ignored citation links with unknown paper IDs: "
                    + unknown);
        }

        split(d, new Random(SEED));

        return d;
    }

    // Stratified 60/20/20 split
    static void split(Data d, Random random) {

        d.train = new boolean[d.n];
        d.val = new boolean[d.n];
        d.test = new boolean[d.n];

        Map<Integer, List<Integer>> groups = new TreeMap<>();

        for (int i = 0; i < d.n; i++) {
            groups.computeIfAbsent(
                    d.y[i], k -> new ArrayList<>()).add(i);
        }

        for (List<Integer> group : groups.values()) {

            Collections.shuffle(group, random);

            int n = group.size();
            int trainCount = Math.max(1, (int) (n * 0.6));
            int valCount = Math.max(1, (int) (n * 0.2));

            if (trainCount + valCount >= n) {
                trainCount = Math.max(1, n - 2);
                valCount = 1;
            }

            for (int k = 0; k < n; k++) {

                int id = group.get(k);

                if (k < trainCount) {
                    d.train[id] = true;
                } else if (k < trainCount + valCount) {
                    d.val[id] = true;
                } else {
                    d.test[id] = true;
                }
            }
        }
    }

    // Three-layer forward propagation
    static Fwd forward(Data d, Model m) {

        Fwd f = new Fwd();

        // Layer 1
        f.ax = aggregate(d, d.x);
        f.u0 = addBias(multiply(f.ax, m.w0), m.b0);
        f.h0 = relu(f.u0);

        // Layer 2
        f.ah0 = aggregate(d, f.h0);
        f.u1 = addBias(multiply(f.ah0, m.w1), m.b1);
        f.h1 = relu(f.u1);

        // Layer 3: output layer
        f.ah1 = aggregate(d, f.h1);
        f.z = addBias(multiply(f.ah1, m.w2), m.b2);
        f.p = softmax(f.z);

        return f;
    }

    // Backpropagation and gradient-descent update
    static void backward(Data d, Model m, Fwd f) {

        int n = d.n;
        int classes = d.c;
        int trainCount = count(d.train);

        double[][] dz = new double[n][classes];

        for (int i = 0; i < n; i++) {
            if (d.train[i]) {
                for (int k = 0; k < classes; k++) {
                    dz[i][k] =
                            (f.p[i][k] - (d.y[i] == k ? 1.0 : 0.0))
                            / trainCount;
                }
            }
        }

        // Layer 3 gradients
        double[][] dw2 = multiply(transpose(f.ah1), dz);
        addL2(dw2, m.w2);

        double[][] gradH1 =
                aggregate(d, multiply(dz, transpose(m.w2)));

        double[][] du1 = reluBackward(gradH1, f.u1);

        // Layer 2 gradients
        double[][] dw1 = multiply(transpose(f.ah0), du1);
        addL2(dw1, m.w1);

        double[][] gradH0 =
                aggregate(d, multiply(du1, transpose(m.w1)));

        double[][] du0 = reluBackward(gradH0, f.u0);

        // Layer 1 gradients
        double[][] dw0 = multiply(transpose(f.ax), du0);
        addL2(dw0, m.w0);

        // Update weights and biases
        update(m.w0, dw0);
        update(m.w1, dw1);
        update(m.w2, dw2);

        update(m.b0, columnSums(du0));
        update(m.b1, columnSums(du1));
        update(m.b2, columnSums(dz));
    }

    // Graph neighbourhood aggregation
    static double[][] aggregate(Data d, double[][] h) {

        double[][] out = new double[d.n][h[0].length];

        for (int i = 0; i < d.n; i++) {

            for (int e = 0; e < d.nbr[i].size(); e++) {

                int j = d.nbr[i].get(e);
                double weight = d.wt[i].get(e);

                for (int k = 0; k < h[j].length; k++) {
                    out[i][k] += weight * h[j][k];
                }
            }
        }

        return out;
    }

    static double[][] reluBackward(double[][] grad, double[][] u) {

        double[][] out = new double[grad.length][grad[0].length];

        for (int i = 0; i < grad.length; i++) {
            for (int j = 0; j < grad[i].length; j++) {
                out[i][j] = u[i][j] > 0 ? grad[i][j] : 0;
            }
        }

        return out;
    }

    static double[][] init(
            int rows, int cols, Random random, double scale) {

        double[][] a = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                a[i][j] = random.nextGaussian() * scale;
            }
        }

        return a;
    }

    static double[][] multiply(double[][] a, double[][] b) {

        double[][] out = new double[a.length][b[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int k = 0; k < b.length; k++) {

                double value = a[i][k];

                if (value == 0) {
                    continue;
                }

                for (int j = 0; j < b[0].length; j++) {
                    out[i][j] += value * b[k][j];
                }
            }
        }

        return out;
    }

    static double[][] transpose(double[][] a) {

        double[][] out = new double[a[0].length][a.length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                out[j][i] = a[i][j];
            }
        }

        return out;
    }

    static double[][] addBias(double[][] a, double[][] b) {

        double[][] out = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                out[i][j] = a[i][j] + b[0][j];
            }
        }

        return out;
    }

    static double[][] relu(double[][] a) {

        double[][] out = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                out[i][j] = Math.max(0, a[i][j]);
            }
        }

        return out;
    }

    static double[][] softmax(double[][] a) {

        double[][] out = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {

            double max = a[i][0];

            for (int j = 1; j < a[i].length; j++) {
                max = Math.max(max, a[i][j]);
            }

            double sum = 0;

            for (int j = 0; j < a[i].length; j++) {
                out[i][j] = Math.exp(a[i][j] - max);
                sum += out[i][j];
            }

            for (int j = 0; j < a[i].length; j++) {
                out[i][j] /= sum;
            }
        }

        return out;
    }

    static double loss(Data d, double[][] p, boolean[] mask) {

        double sum = 0;
        int n = 0;

        for (int i = 0; i < d.n; i++) {
            if (mask[i]) {
                sum -= Math.log(Math.max(1e-12, p[i][d.y[i]]));
                n++;
            }
        }

        return n == 0 ? 0 : sum / n;
    }

    static double accuracy(Data d, double[][] p, boolean[] mask) {

        int correct = 0;
        int total = 0;

        for (int i = 0; i < d.n; i++) {
            if (mask[i]) {
                if (argmax(p[i]) == d.y[i]) {
                    correct++;
                }
                total++;
            }
        }

        return total == 0 ? 0 : (double) correct / total;
    }

    static double[][] confusionMatrix(
            Data d, double[][] p, boolean[] mask) {

        double[][] cm = new double[d.c][d.c];

        for (int i = 0; i < d.n; i++) {
            if (mask[i]) {
                cm[d.y[i]][argmax(p[i])]++;
            }
        }

        return cm;
    }

    static double precision(double tp, double fp) {
        return tp + fp == 0 ? 0 : tp / (tp + fp);
    }

    static double recall(double tp, double fn) {
        return tp + fn == 0 ? 0 : tp / (tp + fn);
    }

    static double f1(double precision, double recall) {
        return precision + recall == 0
                ? 0
                : 2 * precision * recall / (precision + recall);
    }

    static double macroPrecision(
            Data d, double[][] p, boolean[] mask) {

        double[][] cm = confusionMatrix(d, p, mask);
        double sum = 0;

        for (int c = 0; c < d.c; c++) {

            double tp = cm[c][c];
            double fp = 0;

            for (int r = 0; r < d.c; r++) {
                if (r != c) {
                    fp += cm[r][c];
                }
            }

            sum += precision(tp, fp);
        }

        return sum / d.c;
    }

    static double macroRecall(
            Data d, double[][] p, boolean[] mask) {

        double[][] cm = confusionMatrix(d, p, mask);
        double sum = 0;

        for (int c = 0; c < d.c; c++) {

            double tp = cm[c][c];
            double fn = 0;

            for (int col = 0; col < d.c; col++) {
                if (col != c) {
                    fn += cm[c][col];
                }
            }

            sum += recall(tp, fn);
        }

        return sum / d.c;
    }

    static double macroF1(
            Data d, double[][] p, boolean[] mask) {

        double[][] cm = confusionMatrix(d, p, mask);
        double sum = 0;

        for (int c = 0; c < d.c; c++) {

            double tp = cm[c][c];
            double fp = 0;
            double fn = 0;

            for (int r = 0; r < d.c; r++) {
                if (r != c) {
                    fp += cm[r][c];
                    fn += cm[c][r];
                }
            }

            sum += f1(precision(tp, fp), recall(tp, fn));
        }

        return sum / d.c;
    }

    static void printClassMetrics(Data d, double[][] p) {

        double[][] cm = confusionMatrix(d, p, d.test);

        System.out.println("\nPer-class test metrics:");
        System.out.println("Class\tPrecision\tRecall\t\tF1");

        for (int c = 0; c < d.c; c++) {

            double tp = cm[c][c];
            double fp = 0;
            double fn = 0;

            for (int r = 0; r < d.c; r++) {
                if (r != c) {
                    fp += cm[r][c];
                    fn += cm[c][r];
                }
            }

            double pr = precision(tp, fp);
            double re = recall(tp, fn);

            System.out.printf(Locale.US,
                    "%s\t%.4f\t\t%.4f\t\t%.4f%n",
                    d.names[c], pr, re, f1(pr, re));
        }
    }

    static void printConfusionMatrix(Data d, double[][] p) {

        double[][] cm = confusionMatrix(d, p, d.test);

        System.out.println(
                "\nConfusion matrix (actual rows, predicted columns):");

        System.out.print("Actual/Pred");

        for (String name : d.names) {
            System.out.print("\t" + name);
        }

        System.out.println();

        for (int r = 0; r < d.c; r++) {

            System.out.print(d.names[r]);

            for (int c = 0; c < d.c; c++) {
                System.out.print("\t" + (int) cm[r][c]);
            }

            System.out.println();
        }
    }

    static void savePredictions(Data d, double[][] p)
            throws IOException {

        try (PrintWriter out = new PrintWriter(
                new FileWriter(OUT + "/predictions.csv"))) {

            out.println(
                    "paper_id,actual,predicted,correct_class_probability");

            for (int i = 0; i < d.n; i++) {

                if (!d.test[i]) {
                    continue;
                }

                int pred = argmax(p[i]);

                out.printf(Locale.US, "%s,%s,%s,%.6f%n",
                        csv(d.ids[i]),
                        csv(d.names[d.y[i]]),
                        csv(d.names[pred]),
                        p[i][d.y[i]]);
            }
        }
    }

    static int argmax(double[] a) {

        int best = 0;

        for (int i = 1; i < a.length; i++) {
            if (a[i] > a[best]) {
                best = i;
            }
        }

        return best;
    }

    static int count(boolean[] values) {

        int n = 0;

        for (boolean v : values) {
            if (v) {
                n++;
            }
        }

        return n;
    }

    static int edgeCount(Data d) {

        int n = 0;

        for (List<Integer> list : d.nbr) {
            n += list.size() - 1;
        }

        return n;
    }

    static void addL2(double[][] grad, double[][] weights) {

        for (int i = 0; i < grad.length; i++) {
            for (int j = 0; j < grad[i].length; j++) {
                grad[i][j] += L2 * weights[i][j];
            }
        }
    }

    static void update(double[][] weights, double[][] grad) {

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] -= LR * grad[i][j];
            }
        }
    }

    static double[][] columnSums(double[][] a) {

        double[][] sums = new double[1][a[0].length];

        for (double[] row : a) {
            for (int j = 0; j < row.length; j++) {
                sums[0][j] += row[j];
            }
        }

        return sums;
    }

    static Model copy(Model m) {

        Model q = new Model(
                m.w0.length,
                m.w0[0].length,
                m.w1[0].length,
                m.w2[0].length,
                new Random(1));

        q.w0 = copyMatrix(m.w0);
        q.w1 = copyMatrix(m.w1);
        q.w2 = copyMatrix(m.w2);

        q.b0 = copyMatrix(m.b0);
        q.b1 = copyMatrix(m.b1);
        q.b2 = copyMatrix(m.b2);

        return q;
    }

    static double[][] copyMatrix(double[][] a) {

        double[][] b = new double[a.length][];

        for (int i = 0; i < a.length; i++) {
            b[i] = Arrays.copyOf(a[i], a[i].length);
        }

        return b;
    }

    static String csv(String s) {

        if (s.contains(",") || s.contains("\"")) {
            return "\"" + s.replace("\"", "\"\"") + "\"";
        }

        return s;
    }
}