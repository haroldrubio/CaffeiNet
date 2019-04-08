import java.util.List;
public abstract class CompNode extends Node{
    protected OneVarFunction f1;
    protected TwoVarFunction f2;
    protected OneVarDerivative[] derivatives1;
    protected TwoVarDerivative[] derivatives2;
    public CompNode(List<Node> children, List<Node> parents) {
        super(children, parents);
    }
    abstract void forward(double[][] input);
    abstract void backward(double[][] loss);
}
