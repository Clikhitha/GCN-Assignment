
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class DatasetLoader {

    public static void main(String[] args) throws IOException {

        String contentPath = "dataset/cora/cora.content";
        String citesPath = "dataset/cora/cora.cites";

        Map<String, double[]> features = new LinkedHashMap<>();
        Map<String, String> labels = new LinkedHashMap<>();
        List<String[]> edges = new ArrayList<>();

        // Read paper features and class labels
        try (BufferedReader br =
                     new BufferedReader(new FileReader(contentPath))) {

            String line;

            while ((line = br.readLine()) != null) {

                String[] values = line.trim().split("\\s+");

                String paperId = values[0];
                int featureCount = values.length - 2;

                double[] paperFeatures = new double[featureCount];

                for (int i = 1; i < values.length - 1; i++) {
                    paperFeatures[i - 1] =
                            Double.parseDouble(values[i]);
                }

                String label = values[values.length - 1];

                features.put(paperId, paperFeatures);
                labels.put(paperId, label);
            }
        }

        // Read citation edges
        try (BufferedReader br =
                     new BufferedReader(new FileReader(citesPath))) {

            String line;

            while ((line = br.readLine()) != null) {

                String[] values = line.trim().split("\\s+");

                if (values.length >= 2) {
                    edges.add(new String[]{
                            values[0], values[1]
                    });
                }
            }
        }

        // Count distinct classes
        Set<String> classes = new HashSet<>(labels.values());

        System.out.println("========== CORA DATASET ==========");
        System.out.println("Number of papers: " + features.size());
        System.out.println("Features per paper: "
                + features.values().iterator().next().length);
        System.out.println("Citation connections: " + edges.size());
        System.out.println("Number of classes: " + classes.size());

        System.out.println("==================================");
    }
}



