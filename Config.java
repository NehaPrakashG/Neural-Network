import neural.network.Hidden_Layer_Conf;
import neural.network.NeuralNetwork_Config;

import java.util.ArrayList;

public class Config {
    private ArrayList<NeuralNetwork_Config> networkConfigs;
    private Integer Input_Count;
    private Integer Output_Count;
    public Config() {
    	
        this.networkConfigs = networkConfigs;
        this.networkConfigs = new ArrayList<NeuralNetwork_Config>(1);
        this.Test_Config_new();
    }

  
    public ArrayList<NeuralNetwork_Config> getNetworkConfigs() {
        return networkConfigs;
    }

    private void Test_Config_new() {

        Hidden_Layer_Conf H_Config1 = new Hidden_Layer_Conf(150, 's');
        H_Config1.setThreshold(0.6d);

        Hidden_Layer_Conf H_Config2 = new Hidden_Layer_Conf(100, 's');
        H_Config2.setThreshold(0.6d);

        ArrayList<Hidden_Layer_Conf> H_Config = new ArrayList<Hidden_Layer_Conf>();

        H_Config.add(H_Config1);
//        hConfigs.add(hConfig2);

        NeuralNetwork_Config config1 = new NeuralNetwork_Config(H_Config);
        config1.setInputLayerThreshold(0.6d);
        config1.setOutputLayerThreshold(0.6d);
        config1.setOutputLayerTransformFunction('x');
        config1.setLearningRate(0.5);
        config1.setNumberOfEpochs(100);
        this.networkConfigs.add(config1);
    }
}
