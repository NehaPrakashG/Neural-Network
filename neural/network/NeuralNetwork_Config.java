package neural.network;

import java.util.ArrayList;

public class NeuralNetwork_Config {
    private char inputLayer_Function_Transform;
    private double inputLayer_Threshold = 0D;
    private char outputLayer_Function_Transform;
    private double outputLayer_Threshold = 0D;
    private ArrayList<Hidden_Layer_Conf> hiddenLayer_Configs;

    private double learningRate = 0.4D;

    private Integer numberOfEpochs = 2;

    public Integer getNumberOfEpochs() {
        return numberOfEpochs;
    }

    public void setNumberOfEpochs(Integer numberOfEpochs) {
        this.numberOfEpochs = numberOfEpochs;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }


    public double getInputLayerThreshold() {
        return inputLayer_Threshold;
    }

    public void setInputLayerThreshold(double inputLayerThreshold) {
        this.inputLayer_Threshold = inputLayerThreshold;
    }

    public double getOutputLayerThreshold() {
        return outputLayer_Threshold;
    }

    public void setOutputLayerThreshold(double outputLayerThreshold) {
        this.outputLayer_Threshold = outputLayerThreshold;
    }


    public char getInputLayerTransformFunction() {
        return inputLayer_Function_Transform;
    }

    public void setInputLayerTransformFunction(char inputLayerTransformFunction) {
        this.inputLayer_Function_Transform = inputLayerTransformFunction;
    }

    public char getOutputLayerTransformFunction() {
        return outputLayer_Function_Transform;
    }

    public void setOutputLayerTransformFunction(char outputLayerTransformFunction) {
        this.outputLayer_Function_Transform = outputLayerTransformFunction;
    }

    public NeuralNetwork_Config(ArrayList<Hidden_Layer_Conf> hiddenLayerConfigs) {
        this.hiddenLayer_Configs = hiddenLayerConfigs;
    }

    public ArrayList<Hidden_Layer_Conf> getHiddenLayerConfigs() {
        return hiddenLayer_Configs;
    }

    public void setHiddenLayerConfigs(ArrayList<Hidden_Layer_Conf> hiddenLayerConfigs) {
        this.hiddenLayer_Configs = hiddenLayerConfigs;
    }

    private String getHiddenLayersString() {
        int layerNumber = 0;
        String result = "";
        for (Hidden_Layer_Conf config : this.hiddenLayer_Configs) {
            result += "Hidden layer " + (++layerNumber) + "\n" + config.toString() + "\n";
        }
        return result;
    }

    @Override
    public String toString() {
        return  "InputLayerTransformFunction=" + inputLayer_Function_Transform +
                "\nInputLayerThreshold=" + inputLayer_Threshold +
                "\nOutputLayerTransformFunction=" + outputLayer_Function_Transform +
                "\nOutputLayerThreshold=" + outputLayer_Threshold +
                "\nLearningRate=" + learningRate +
                "\nIterations=" + numberOfEpochs +
                "\nHidden layers config:\n" + this.getHiddenLayersString() +
                '}';
    }
}
