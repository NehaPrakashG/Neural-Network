import neural.network.NeuralNetwork;
import neural.network.NeuralNetwork_Config;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import java.util.logging.Level;
import java.util.logging.Logger;

public class MainFile {

	private static final String TRAINING_DATA= "C:/Users/gaikw/Desktop/Sem1/AOS/emnist-balanced-trainFor2Class.arff";
    private static final String TESTING_DATA= "C:/Users/gaikw/Desktop/Sem1/AOS/emnist-balanced-testFor2Class.arff";

    public static void main(String[] args) throws Exception {


        Filter filterparam = new Normalize();
        
        System.out.println("Loading train data");
        long startTime = System.nanoTime();
        Instances Data_Files = loadData_Files(MainFile.TRAINING_DATA);
        
        Data_Files.setClassIndex(Data_Files.numAttributes() - 1);
        // DIVIDING Data_Files TO TRAIN 80% AND TEST 20%
        int train_data_size = (int) Math.round(Data_Files.numInstances() * 0.8);
        int test_data_Size = Data_Files.numInstances() - train_data_size;
        Data_Files.randomize(new Debug.Random(1));
        filterparam.setInputFormat(Data_Files);
        Instances normalizDataFiles = Filter.useFilter(Data_Files, filterparam);

        Instances normaliz_Train_Set = new Instances(normalizDataFiles, 0, train_data_size);
        Instances normaliz_Valid_Set = new Instances(normalizDataFiles, train_data_size, test_data_Size);

        Data_Files.delete();
        long endTime = System.nanoTime();

        Config testConfigs = new Config();
        System.out.println("Training : " + normaliz_Train_Set.size() + "\nValidation : " + normaliz_Valid_Set.size() + "\nTime taken: " + (endTime - startTime) / 1000000 + " milliseconds");

        NeuralNetwork_Config testConfig = testConfigs.getNetworkConfigs().get(0);

        NeuralNetwork Ntw = new NeuralNetwork(testConfig, normaliz_Train_Set, normaliz_Valid_Set);
        Ntw.train();
        System.out.println(Ntw.toString());

        System.out.println("Loading testing data...");
        startTime = System.nanoTime();
        Instances testingSet = loadData_Files(MainFile.TESTING_DATA);
        
        testingSet.randomize(new Debug.Random(1));
        filterparam.setInputFormat(testingSet);
        Instances norm_Test_set = filterparam.useFilter(testingSet, filterparam);
        norm_Test_set.setClassIndex(norm_Test_set.numAttributes() - 1);
        testingSet.delete();
        endTime = System.nanoTime();
        System.out.println("Testing set : " + norm_Test_set.size() + "\nTime taken by them: " + (endTime - startTime) / 1000000 + " milliseconds");
        System.out.println("Predicting");
        for (Instance ins : norm_Test_set) {
            Ntw.predictClass(ins);
        }
        // CORRECT PREDICTION
        long Correct_Pred_Count = Ntw.getCountOfCorrectPredictions();
        System.out.println("Accuracy od data : " + (Double.parseDouble(Correct_Pred_Count + "") / norm_Test_set.size()) + " (" + Correct_Pred_Count + "/" + norm_Test_set.size() + ")");
    }

    private static Instances loadData_Files(String Path) {
        Instances file = null;
        try {
        	//CONVERTING DATA USING WEKA
        	file = ConverterUtils.DataSource.read(Path);
            if (file.classIndex() == -1) {
            	file.setClassIndex(file.numAttributes() - 1);
            }
        } catch (Exception ex) {
            Logger.getAnonymousLogger().log(Level.SEVERE, null, ex);
        }

        return file;
    }
}
