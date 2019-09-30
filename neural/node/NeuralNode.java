package neural.node;


import neural.network.*;

import java.util.ArrayList;


public class NeuralNode {
    private static Long counter = 0L;
    private Long id;
    private Neural_Layer layer;

    private ArrayList<NeuralConnection> incomingConnections;
    private ArrayList<NeuralConnection> outgoingConnections;

    private double activation = 0d;
    private double thresholdValue = 0d;
    private double output = 0d;
    private double target;

    private double errorValue = 0d;

    private char Function_transform;

    public double getactivation() {
        return activation;
    }

    public void setactivation(double activation) {
        this.activation = activation;
    }

    public double gettarget() {
        return target;
    }

    public void settarget(double target) {
        this.target = target;
    }

    public double getErrorValue() {
        return errorValue;
    }

    public void setErrorValue(double errorValue) {
        this.errorValue = errorValue;
    }

    public char getFunction_transform() {
        return Function_transform;
    }

    public void setFunction_transform(char Function_transform) {
        this.Function_transform = Function_transform;
    }

    public void StratNeuron() {
        if (null == this.incomingConnections) {
            return;
        }
        this.activation = 0d;
        long norm = 0l;
        for (NeuralConnection connection : this.incomingConnections) {
            if (connection.getNode_A().getoutput() > 0) {
            	norm++;
            }
            this.activation += connection.getNode_B().getoutput() * connection.getNode_A().getoutput();
        }
        this.activation = (this.activation / norm) - this.thresholdValue;
  }

    public void transform() {
        if (null == this.incomingConnections) {
            return;
        }
        switch (this.Function_transform) {
            case 's':
                this.output = this.transform_Sigmoid(this.activation);
                break;
            case 'x':
                this.output = this.transform_Softmax(this.activation);
                break;
            default:
                this.output = this.activation;
        }
    }

    public void backPropagate_Error() {
        switch (this.Function_transform) {
            case 's':
                this.errorValue = this.getError(this.getDerivativeSigmoid(this.output));
                break;
            case 'x':
                this.errorValue = this.getError(this.getDerivativeSoftmax(this.output));
                break;
            default:
                this.errorValue = this.getError(1d);
        }
    }

    public double getThresholdValue() {
        return thresholdValue;
    }

    public void setThresholdValue(double thresholdValue) {
        this.thresholdValue = thresholdValue;
    }

    public double getoutput() {
        return output;
    }

    public void setoutput(double output) {
        this.output = output;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Neural_Layer getLayer() {
        return layer;
    }

    public void setLayer(Neural_Layer layer) {
        this.layer = layer;
    }

    public ArrayList<NeuralConnection> getIncomingConnections() {
        return incomingConnections;
    }

    public void setIncomingConnections(ArrayList<NeuralConnection> incomingConnections) {
        this.incomingConnections = incomingConnections;
    }

    public ArrayList<NeuralConnection> getOutgoingConnections() {
        return outgoingConnections;
    }

    public void setOutgoingConnections(ArrayList<NeuralConnection> outgoingConnections) {
        this.outgoingConnections = outgoingConnections;
    }

    public NeuralNode(Neural_Layer layer) {
        this.layer = layer;
        this.id = ++NeuralNode.counter;
        this.initializeNeuralNode();
    }

    public NeuralNode(Neural_Layer layer, double target) {
        this.layer = layer;
        this.id = ++NeuralNode.counter;
        this.target = target;
        this.initializeNeuralNode();
    }

    public void updateValue(double delta) {
        this.output += delta;
    }

    private void initializeNeuralNode() {
        this.incomingConnections = new ArrayList<NeuralConnection>();
        this.outgoingConnections = new ArrayList<NeuralConnection>();
        this.Function_transform = layer.getFunction_Transform();
        this.thresholdValue = layer.getThreshold();
    }

    private double getError(double delta) {
        double currentErrorValue = 0d;
        if (this.outgoingConnections.isEmpty()) {
            currentErrorValue = (this.target - this.output) * delta;
        } else {
            for (NeuralConnection connection : this.outgoingConnections) {
                NeuralNode nodeConnectedTo = connection.getNode_B();
                currentErrorValue += connection.getWt() * nodeConnectedTo.getErrorValue() * delta;
            }
        }
        return currentErrorValue;
    }

    private double transform_Softmax(double value) {
        double totalLayerInput = 0d;

        for (NeuralNode node : this.layer.getnn()) {
            totalLayerInput += Math.exp(node.getactivation());
        }

        double output = Math.exp(value) / totalLayerInput;
        return output;
    }

    private double getDerivativeSoftmax(double value) {
        return value * (1d - value);
    }

    private double transform_Sigmoid(double value) {

        return 1 / (1 + (Math.exp(-1 * value)));
    }

    private double getDerivativeSigmoid(double value) {
        return value * (1d - value);
    }
}
