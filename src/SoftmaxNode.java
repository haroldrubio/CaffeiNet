import java.util.List;
public class SoftmaxNode extends CompNode{
    public SoftmaxNode(List<Node> children, List<Node> parents){
        super(children, parents);
        this.f1 = new OneVarFunction() {
            @Override
            public double[][] f(double[][] input) {
                return softmaxRows(input);
            }
        };
        this.derivatives1 = new OneVarDerivative[]{
                new OneVarDerivative() {
                    @Override
                    public double[][] deriv(double[][] loss) {
                        return loss;
                    }
                }
        };
    }

    public void forward(double[][] input){
        this.hiddenState = this.f1.f(input);
        this.prevHidden = input;
        super.passToChildren(hiddenState);
    }

    public void backward(double[][] loss){
        super.passToParents(loss);
    }

    private double[][] softmaxRows(double[][] input){
        int batchSize = input.length;
        double[][] softmaxProb = new double[batchSize][];
        for(int i = 0; i < batchSize; i++){
            softmaxProb[i] = Stats.softmax(input[i]);
        }
        return softmaxProb;
    }

}
