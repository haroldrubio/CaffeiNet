import java.util.List;

/**
 * Performs a softmax calculation on the hidden state of the previous layer
 */
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

    /**
     * Passes the gradient unchanged
     * No modification is made to the gradient since the cross entropy loss is directly calculated
     * with respect to the hidden state
     * @param loss A batch of gradients
     */
    public void backward(double[][] loss){
        super.passToParents(loss);
    }

    /**
     * Given a batch of hidden state vectors, perform an independent softmax normalization
     * on each of the rows
     * @param input A double matrix
     * @return A double matrix with rows of probability
     */
    private double[][] softmaxRows(double[][] input){
        int batchSize = input.length;
        double[][] softmaxProb = new double[batchSize][];
        for(int i = 0; i < batchSize; i++){
            softmaxProb[i] = Stats.softmax(input[i]);
        }
        return softmaxProb;
    }

}
