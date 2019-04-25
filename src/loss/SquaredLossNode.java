import java.util.List;
/**
 * A terminating node in the computational graph that computes squared error of the form
 * 1/2(y_pred - y_corr)^2 for the task of regression
 */
public class SquaredLossNode extends LossNode{
    public SquaredLossNode(List<Node> children, List<Node> parents){
        super(children, parents);
        this.f1 = new OneVarFunction() {
            @Override
            public double[][] f(double[][] input) {
                return squaredLoss(input);
            }
        };
        this.derivatives1 = new OneVarDerivative[]{
                new OneVarDerivative() {
                    @Override
                    public double[][] deriv(double[][] loss) {
                        return squaredLossGrad(loss);
                    }
                }
        };
    }

    /**
     * Computes the loss by taking half of the squared L2 norm for every example in the batch
     * Returns an nx1 vector of the squared loss
     * @param correct A matrix of the correct points
     * @return A double matrix
     */
    private double[][] squaredLoss(double[][] correct){
        int batchSize = correct.length;
        double[][] totalLoss = new double[batchSize][1];
        double[] predictedVector, correctVector;
        for(int i = 0; i < batchSize; i++){
            predictedVector = predictions[i];
            correctVector = correct[i];
            totalLoss[i][0] = MatrixOps.l2Norm(MatrixOps.difference(predictedVector, correctVector));
            totalLoss[i][0] = 0.5*Math.pow(totalLoss[i][0], 2);
        }
        return totalLoss;
    }

    private double[][] squaredLossGrad(double[][] correct){
        int batchSize = correct.length, outputDims = correct[0].length;
        double[][] gradient = new double[batchSize][outputDims];
        double[] predictedVector, correctVector;
        for(int i = 0; i < batchSize; i++){
            predictedVector = predictions[i];
            correctVector = correct[i];
            gradient[i] = MatrixOps.difference(predictedVector, correctVector);
        }
        return gradient;
    }

}
