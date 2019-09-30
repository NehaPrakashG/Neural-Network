package neural.network;

import neural.network.NeuralNetwork_Config;
import neural.node.NeuralNode;
import java.util.ArrayList;

public class Neural_Layer {
    private static Long Counterer = 0l;
    private Long id;
    private Integer Node_No;
    private char Function_Transform;
    private ArrayList<NeuralNode> nn;
    private double threshold = 0D;
    private double learnRate = 0D;

    public Neural_Layer(Integer Node_No, char Transform, double threshold, double learnRate) {
        this.id = ++Neural_Layer.Counterer;
        this.Node_No = Node_No;
        this.Function_Transform = Transform;
        this.threshold = threshold;
        this.learnRate = learnRate;
        this.initializenn();
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Integer getNode_No() {
        return Node_No;
    }

    public void setNode_No(Integer Node_No) {
        this.Node_No = Node_No;
    }

    public ArrayList<NeuralNode> getnn() {
        return nn;
    }

    public void setnn(ArrayList<NeuralNode> nn) {
        this.nn = nn;
    }

    private void initializenn() {
        if (this.Node_No <=0) {
            return;
        }
        this.nn = new ArrayList<NeuralNode>(this.Node_No);
        for (int i = 0; i < this.Node_No; i++) {
            NeuralNode node = new NeuralNode(this);
            node.setThresholdValue(this.threshold);
            nn.add(node);
        }
    }

    public char getFunction_Transform() {
        return Function_Transform;
    }

    public void setFunction_Transform(char Transform) {
        this.Function_Transform = Transform;
    }

    public double getThreshold() {
        return threshold;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }
}
