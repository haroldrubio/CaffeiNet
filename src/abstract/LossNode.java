import java.util.List;
public abstract class LossNode extends CompNode{
    protected double[][] predictions;
    protected double[][] batchLoss;
    public LossNode(List<Node> children, List<Node> parents){
        super(children, parents);
    }
    public void forward(double[][] predicted){
        predictions = predicted;
        batchLoss = this.f1.f(predicted);
    }
    public void backward(double[][] correct){
        CompNode nextComp;
        List<Node> currentParents = this.getParents();
        double[][] nextGradient = null;
        //Send the signal backwards
        if(correct != null)
            nextGradient = this.derivatives1[0].deriv(correct);
        if(currentParents != null)
            for(Node n: currentParents)
                if(n instanceof CompNode) {
                    nextComp = (CompNode) n;
                    nextComp.backward(nextGradient);
                }
    }
}
