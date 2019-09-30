package neural.network;

public class Hidden_Layer_Conf {

    private double threshold = 0D;
    private Integer Hidden_Units;
    private char Function_Transform;
    
    public double getThreshold() {
        return threshold;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    public char getFunction_Transform() {
        return Function_Transform;
    }

    public void setFunction_Transform(char Function_Transform) {
        this.Function_Transform = Function_Transform;
    }

    public Integer getHidden_Units() {
        return Hidden_Units;
    }

    public void setHidden_Units(Integer Hidden_Units) {
        this.Hidden_Units = Hidden_Units;
    }

    public Hidden_Layer_Conf(Integer Hidden_Units, char Function_Transform) {
        this.Hidden_Units = Hidden_Units;
        this.Function_Transform = Function_Transform;
    }

    @Override
    public String toString() {
        return "Hidden_Units=" + Hidden_Units +
                "\nFunction_Transform=" + Function_Transform +
                "\nThreshold=" + threshold;
    }
}

