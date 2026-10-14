import java.io.*;

import java.util.*;



public class TwoLayerGCN {



    // Dataset and training configuration

    static final String CONTENT_PATH = "dataset/cora/cora.content";

    static final String CITES_PATH = "dataset/cora/cora.cites";



    static final int HIDDEN = 32;

    static final int EPOCHS = 300;

    static final double LEARNING_RATE = 0.005;

    static final double L2 = 0.0005;

    static final long SEED = 42;



    static class GraphData {

        int n, features, classes;

        int[][] x;

        int[] y;

        String[] ids;

        String[] classNames;

        List<Integer>[] neighbors;

        List<Double>[] weights;

        boolean[] train, val, test;

    }



    static class Model {

        double[][] w0, w1;

        double[][] b0, b1;



        Model(int input, int hidden, int output, Random random) {

            w0 = new double[input][hidden];

            w1 = new double[hidden][output];

            b0 = new double[1][hidden];

            b1 = new double[1][output];



            double s0 = Math.sqrt(2.0 / input);

            double s1 = Math.sqrt(2.0 / hidden);



            for (int i = 0; i < input; i++)

                for (int j = 0; j < hidden; j++)

                    w0[i][j] = random.nextGaussian() * s0;



            for (int i = 0; i < hidden; i++)

                for (int j = 0; j < output; j++)

                    w1[i][j] = random.nextGaussian() * s1;

        }

    }



    static class Forward {

        double[][] h;

        double[][] probabilities;

    }



    public static void main(String[] args) throws Exception {

        GraphData data = loadData();



        System.out.println("========== CORA DATASET ==========");

        System.out.println("Nodes: " + data.n);

        System.out.println("Features per node: " + data.features);

        System.out.println("Classes: " + data.classes);

        System.out.println("Undirected graph edges (excluding self-loops): "

                + countEdges(data) / 2);

        System.out.println("Training nodes: " + count(data.train));

        System.out.println("Validation nodes: " + count(data.val));

        System.out.println("Testing nodes: " + count(data.test));



        Random random = new Random(SEED);

        Model model = new Model(

                data.features, HIDDEN, data.classes, random);



        double bestF1 = -1;

        double[][] bestW0 = null, bestW1 = null;

        double[][] bestB0 = null, bestB1 = null;



        try (PrintWriter log = new PrintWriter(

                new FileWriter("results/training.csv"))) {



            log.println("epoch,train_loss,val_accuracy,val_macro_f1");



            for (int epoch = 1; epoch <= EPOCHS; epoch++) {

                Forward f = forward(data, model);

                double loss = loss(data, f.probabilities, data.train);



                backwardAndUpdate(data, model, f, random);



                Forward after = forward(data, model);

                double valAcc = accuracy(

                        data, after.probabilities, data.val);

                double valF1 = macroF1(

                        data, after.probabilities, data.val);



                log.printf(Locale.US, "%d,%.6f,%.6f,%.6f%n",

                        epoch, loss, valAcc, valF1);



                if (valF1 > bestF1) {

                    bestF1 = valF1;

                    bestW0 = copy(model.w0);

                    bestW1 = copy(model.w1);

                    bestB0 = copy(model.b0);

                    bestB1 = copy(model.b1);

                }



                if (epoch == 1 || epoch % 20 == 0) {

                    System.out.printf(Locale.US,

                            "Epoch %3d | Loss %.5f | Val Acc %.4f"

                                    + " | Val Macro-F1 %.4f%n",

                            epoch, loss, valAcc, valF1);

                }

            }

        }



        model.w0 = bestW0;

        model.w1 = bestW1;

        model.b0 = bestB0;

        model.b1 = bestB1;



        Forward finalPass = forward(data, model);



        System.out.println("\n========== TEST RESULTS ==========");

        System.out.printf(Locale.US, "Accuracy : %.4f%n",

                accuracy(data, finalPass.probabilities, data.test));

        System.out.printf(Locale.US, "Precision: %.4f%n",

                macroPrecision(data, finalPass.probabilities, data.test));

        System.out.printf(Locale.US, "Recall   : %.4f%n",

                macroRecall(data, finalPass.probabilities, data.test));

        System.out.printf(Locale.US, "Macro-F1 : %.4f%n",

                macroF1(data, finalPass.probabilities, data.test));



        printClassMetrics(data, finalPass.probabilities);

        printConfusionMatrix(data, finalPass.probabilities);

        savePredictions(data, finalPass.probabilities);



        System.out.println("\nResults saved in the results folder.");

    }



    // Read the Cora feature and citation files

    static GraphData loadData() throws IOException {

        GraphData d = new GraphData();



        List<String> ids = new ArrayList<>();

        List<int[]> featureRows = new ArrayList<>();

        List<String> labelRows = new ArrayList<>();



        Map<String, Integer> index = new HashMap<>();

        TreeSet<String> labelSet = new TreeSet<>();



        try (BufferedReader br = new BufferedReader(

                new FileReader(CONTENT_PATH))) {

            String line;



            while ((line = br.readLine()) != null) {

                String[] p = line.trim().split("\\s+");

                if (p.length < 3) continue;



                int id = ids.size();

                ids.add(p[0]);

                index.put(p[0], id);



                int[] features = new int[p.length - 2];

                for (int j = 1; j < p.length - 1; j++)

                    features[j - 1] = Integer.parseInt(p[j]);



                featureRows.add(features);

                labelRows.add(p[p.length - 1]);

                labelSet.add(p[p.length - 1]);

            }

        }



        d.n = ids.size();

        d.features = featureRows.get(0).length;

        d.ids = ids.toArray(new String[0]);

        d.classNames = labelSet.toArray(new String[0]);

        d.classes = d.classNames.length;

        d.x = featureRows.toArray(new int[0][]);

        d.y = new int[d.n];



        Map<String, Integer> labelIndex = new HashMap<>();

        for (int i = 0; i < d.classes; i++)

            labelIndex.put(d.classNames[i], i);



        for (int i = 0; i < d.n; i++)

            d.y[i] = labelIndex.get(labelRows.get(i));



        // Construct an undirected graph and add self-loops.

        List<Set<Integer>> adjacency = new ArrayList<>();

        for (int i = 0; i < d.n; i++) {

            adjacency.add(new HashSet<>());

            adjacency.get(i).add(i);

        }



        try (BufferedReader br = new BufferedReader(

                new FileReader(CITES_PATH))) {

            String line;



            while ((line = br.readLine()) != null) {

                String[] p = line.trim().split("\\s+");

                if (p.length < 2) continue;



                Integer u = index.get(p[0]);

                Integer v = index.get(p[1]);



                // Ignore references to IDs absent from the content file.

                if (u != null && v != null) {

                    adjacency.get(u).add(v);

                    adjacency.get(v).add(u);

                }

            }

        }



        // Symmetric normalization:

        // A_hat = D^(-1/2) * (A + I) * D^(-1/2)

        d.neighbors = new List[d.n];

        d.weights = new List[d.n];



        double[] degree = new double[d.n];

        for (int i = 0; i < d.n; i++)

            degree[i] = adjacency.get(i).size();



        for (int i = 0; i < d.n; i++) {

            d.neighbors[i] = new ArrayList<>();

            d.weights[i] = new ArrayList<>();



            for (int j : adjacency.get(i)) {

                d.neighbors[i].add(j);

                d.weights[i].add(

                        1.0 / Math.sqrt(degree[i] * degree[j]));

            }

        }



        makeSplits(d, new Random(SEED));

        return d;

    }



    // Stratified split: 60% train, 20% validation, 20% test.

    static void makeSplits(GraphData d, Random random) {

        List<List<Integer>> byClass = new ArrayList<>();

        for (int c = 0; c < d.classes; c++)

            byClass.add(new ArrayList<>());



        for (int i = 0; i < d.n; i++)

            byClass.get(d.y[i]).add(i);



        d.train = new boolean[d.n];

        d.val = new boolean[d.n];

        d.test = new boolean[d.n];



        for (List<Integer> group : byClass) {

            Collections.shuffle(group, random);

            int nTrain = (int) (group.size() * 0.60);

            int nVal = (int) (group.size() * 0.20);



            for (int i = 0; i < group.size(); i++) {

                int node = group.get(i);

                if (i < nTrain) d.train[node] = true;

                else if (i < nTrain + nVal) d.val[node] = true;

                else d.test[node] = true;

            }

        }

    }



    // Forward pass:

    // H = ReLU(A_hat * X * W0 + b0)

    // P = Softmax(A_hat * H * W1 + b1)

    static Forward forward(GraphData d, Model m) {

        double[][] xw = new double[d.n][HIDDEN];



        for (int i = 0; i < d.n; i++) {

            for (int f = 0; f < d.features; f++) {

                if (d.x[i][f] == 0) continue;

                for (int h = 0; h < HIDDEN; h++)

                    xw[i][h] += d.x[i][f] * m.w0[f][h];

            }

        }



        double[][] h = propagate(d, xw);



        for (int i = 0; i < d.n; i++)

            for (int j = 0; j < HIDDEN; j++)

                h[i][j] = Math.max(0, h[i][j] + m.b0[0][j]);



        double[][] hw = multiply(h, m.w1);



        double[][] logits = propagate(d, hw);

        for (int i = 0; i < d.n; i++)

            for (int c = 0; c < d.classes; c++)

                logits[i][c] += m.b1[0][c];



        Forward result = new Forward();

        result.h = h;

        result.probabilities = softmax(logits);

        return result;

    }



    // Sparse graph propagation: A_hat * matrix

    static double[][] propagate(GraphData d, double[][] matrix) {

        int cols = matrix[0].length;

        double[][] result = new double[d.n][cols];



        for (int i = 0; i < d.n; i++) {

            for (int k = 0; k < d.neighbors[i].size(); k++) {

                int j = d.neighbors[i].get(k);

                double weight = d.weights[i].get(k);



                for (int c = 0; c < cols; c++)

                    result[i][c] += weight * matrix[j][c];

            }

        }

        return result;

    }



    // Backpropagation and gradient descent

    static void backwardAndUpdate(

            GraphData d, Model m, Forward f, Random random) {



        double[][] gradLogits = new double[d.n][d.classes];

        int trainCount = count(d.train);



        for (int i = 0; i < d.n; i++) {

            if (!d.train[i]) continue;

            for (int c = 0; c < d.classes; c++) {

                gradLogits[i][c] =

                        (f.probabilities[i][c]

                                - (d.y[i] == c ? 1.0 : 0.0))

                                / trainCount;

            }

        }



        // Output layer: logits = A_hat * H * W1 + b1

        double[][] aGrad = propagate(d, gradLogits);

        double[][] gradW1 = multiply(transpose(f.h), aGrad);

        double[][] gradB1 = columnSums(gradLogits);

        double[][] gradH = multiply(aGrad, transpose(m.w1));



        // Backpropagate through ReLU

        for (int i = 0; i < d.n; i++)

            for (int h = 0; h < HIDDEN; h++)

                if (f.h[i][h] <= 0)

                    gradH[i][h] = 0;



        // Hidden layer: H = ReLU(A_hat * X * W0 + b0)

        double[][] gradPre = propagate(d, gradH);

        double[][] gradW0 = new double[d.features][HIDDEN];



        for (int i = 0; i < d.n; i++) {

            for (int feature = 0; feature < d.features; feature++) {

                if (d.x[i][feature] == 0) continue;

                for (int h = 0; h < HIDDEN; h++)

                    gradW0[feature][h] +=

                            d.x[i][feature] * gradPre[i][h];

            }

        }



        double[][] gradB0 = columnSums(gradH);



        // Weight decay applies to weights, not biases.

        for (int i = 0; i < d.features; i++)

            for (int h = 0; h < HIDDEN; h++)

                gradW0[i][h] += L2 * m.w0[i][h];



        for (int i = 0; i < HIDDEN; i++)

            for (int c = 0; c < d.classes; c++)

                gradW1[i][c] += L2 * m.w1[i][c];



        update(m.w0, gradW0);

        update(m.w1, gradW1);

        update(m.b0, gradB0);

        update(m.b1, gradB1);

    }



    static void update(double[][] weights, double[][] gradients) {

        for (int i = 0; i < weights.length; i++)

            for (int j = 0; j < weights[i].length; j++)

                weights[i][j] -= LEARNING_RATE * gradients[i][j];

    }



    static double loss(

            GraphData d, double[][] p, boolean[] mask) {

        double total = 0;

        int n = count(mask);



        for (int i = 0; i < d.n; i++)

            if (mask[i])

                total -= Math.log(Math.max(

                        1e-12, p[i][d.y[i]]));



        return total / n;

    }



    static double[][] softmax(double[][] logits) {

        double[][] p = new double[logits.length][logits[0].length];



        for (int i = 0; i < logits.length; i++) {

            double max = Arrays.stream(logits[i]).max().orElse(0);

            double sum = 0;



            for (int c = 0; c < logits[i].length; c++) {

                p[i][c] = Math.exp(logits[i][c] - max);

                sum += p[i][c];

            }



            for (int c = 0; c < logits[i].length; c++)

                p[i][c] /= sum;

        }

        return p;

    }



    static int predict(double[] p) {

        int best = 0;

        for (int c = 1; c < p.length; c++)

            if (p[c] > p[best]) best = c;

        return best;

    }



    static double accuracy(

            GraphData d, double[][] p, boolean[] mask) {

        int correct = 0, total = 0;

        for (int i = 0; i < d.n; i++) {

            if (!mask[i]) continue;

            total++;

            if (predict(p[i]) == d.y[i]) correct++;

        }

        return total == 0 ? 0 : (double) correct / total;

    }



    static double[][] confusionMatrix(

            GraphData d, double[][] p, boolean[] mask) {

        double[][] cm = new double[d.classes][d.classes];



        for (int i = 0; i < d.n; i++)

            if (mask[i])

                cm[d.y[i]][predict(p[i])]++;



        return cm;

    }



    static double precision(double tp, double fp) {

        return tp + fp == 0 ? 0 : tp / (tp + fp);

    }



    static double recall(double tp, double fn) {

        return tp + fn == 0 ? 0 : tp / (tp + fn);

    }



    static double f1(double p, double r) {

        return p + r == 0 ? 0 : 2 * p * r / (p + r);

    }



    static double macroPrecision(

            GraphData d, double[][] p, boolean[] mask) {

        double[][] cm = confusionMatrix(d, p, mask);

        double sum = 0;



        for (int c = 0; c < d.classes; c++) {

            double tp = cm[c][c], fp = 0;

            for (int r = 0; r < d.classes; r++)

                if (r != c) fp += cm[r][c];

            sum += precision(tp, fp);

        }

        return sum / d.classes;

    }



    static double macroRecall(

            GraphData d, double[][] p, boolean[] mask) {

        double[][] cm = confusionMatrix(d, p, mask);

        double sum = 0;



        for (int c = 0; c < d.classes; c++) {

            double tp = cm[c][c], fn = 0;

            for (int col = 0; col < d.classes; col++)

                if (col != c) fn += cm[c][col];

            sum += recall(tp, fn);

        }

        return sum / d.classes;

    }



    static double macroF1(

            GraphData d, double[][] p, boolean[] mask) {

        double[][] cm = confusionMatrix(d, p, mask);

        double sum = 0;



        for (int c = 0; c < d.classes; c++) {

            double tp = cm[c][c], fp = 0, fn = 0;



            for (int r = 0; r < d.classes; r++) {

                if (r != c) {

                    fp += cm[r][c];

                    fn += cm[c][r];

                }

            }



            sum += f1(precision(tp, fp), recall(tp, fn));

        }

        return sum / d.classes;

    }



    static void printClassMetrics(

            GraphData d, double[][] p) {

        double[][] cm = confusionMatrix(d, p, d.test);



        System.out.println("\nPer-class test metrics:");

        System.out.println("Class\tPrecision\tRecall\t\tF1");



        for (int c = 0; c < d.classes; c++) {

            double tp = cm[c][c], fp = 0, fn = 0;

            for (int r = 0; r < d.classes; r++) {

                if (r != c) {

                    fp += cm[r][c];

                    fn += cm[c][r];

                }

            }



            System.out.printf(Locale.US, "%s\t%.4f\t\t%.4f\t\t%.4f%n",

                    d.classNames[c],

                    precision(tp, fp),

                    recall(tp, fn),

                    f1(precision(tp, fp), recall(tp, fn)));

        }

    }



    static void printConfusionMatrix(

            GraphData d, double[][] p) {

        double[][] cm = confusionMatrix(d, p, d.test);



        System.out.println("\nConfusion matrix (actual rows, predicted columns):");

        System.out.print("Actual\\Pred");

        for (String name : d.classNames)

            System.out.print("\t" + name);

        System.out.println();



        for (int r = 0; r < d.classes; r++) {

            System.out.print(d.classNames[r]);

            for (int c = 0; c < d.classes; c++)

                System.out.print("\t" + (int) cm[r][c]);

            System.out.println();

        }

    }



    static void savePredictions(

            GraphData d, double[][] p) throws IOException {

        try (PrintWriter out = new PrintWriter(

                new FileWriter("results/predictions.csv"))) {

            out.println("paper_id,actual,predicted,correct_class_probability");



            for (int i = 0; i < d.n; i++) {

                if (!d.test[i]) continue;

                int pred = predict(p[i]);



                out.printf(Locale.US, "%s,%s,%s,%.6f%n",

                        d.ids[i], d.classNames[d.y[i]],

                        d.classNames[pred], p[i][d.y[i]]);

            }

        }

    }



    static int count(boolean[] values) {

        int n = 0;

        for (boolean v : values) if (v) n++;

        return n;

    }



    static int countEdges(GraphData d) {

        int n = 0;

        for (List<Integer> list : d.neighbors)

            n += list.size();

        return n;

    }



    static double[][] multiply(double[][] a, double[][] b) {

        double[][] c = new double[a.length][b[0].length];



        for (int i = 0; i < a.length; i++)

            for (int k = 0; k < b.length; k++) {

                double value = a[i][k];

                if (value == 0) continue;

                for (int j = 0; j < b[0].length; j++)

                    c[i][j] += value * b[k][j];

            }

        return c;

    }



    static double[][] transpose(double[][] a) {

        double[][] t = new double[a[0].length][a.length];

        for (int i = 0; i < a.length; i++)

            for (int j = 0; j < a[i].length; j++)

                t[j][i] = a[i][j];

        return t;

    }



    static double[][] columnSums(double[][] a) {

        double[][] sums = new double[1][a[0].length];

        for (double[] row : a)

            for (int j = 0; j < row.length; j++)

                sums[0][j] += row[j];

        return sums;

    }



    static double[][] copy(double[][] a) {

        double[][] b = new double[a.length][];

        for (int i = 0; i < a.length; i++)

            b[i] = a[i].clone();

        return b;

    }

}
