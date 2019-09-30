package neural.network;

import neural.node.NeuralNode;
import java.util.Random;
public class NeuralConnection {
    private static Long counter = 0l;
    private Long id;
    private NeuralNode Node_A;
    private NeuralNode Node_B;
    private double wt = 0d;
    private double wt_math = Math.random();
    private double lr = 0.3d;

    public NeuralConnection(NeuralNode nodeA, NeuralNode nodeB) {
        this.Node_A = nodeA;
        this.Node_B = nodeB;
        this.id = ++NeuralConnection.counter;
        this.wt = this.wt;
    }
    
    public double getWt() {
        return wt_math;
    }

    public void setWt(double weight) {
        this.wt_math = weight;
    }

    public double getLR() {
        return lr;
        }

    public void setLR(double learnRate) {
        this.lr = learnRate;
    }
    public NeuralNode getNode_B() {
        return Node_B;
    }

    public void setNode_B(NeuralNode nB) {
        this.Node_B = nB;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public NeuralNode getNode_A() {
        return Node_A;
    }

    public void setNode_A(NeuralNode nA) {
        this.Node_A = nA;
    }

  
    public void Update_Wt(int epoch_No) {
        this.wt = this.wt_math;
        this.wt_math += this.lr * this.Node_B.getErrorValue() * this.Node_A.getErrorValue();
    }

    public void revertWt() {
        this.wt = this.wt;
    }
}
