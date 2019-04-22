import java.util.List;
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
    private double[][] squaredLoss(double[][] correct){
        int batchSize = correct.length;
        double[][] totalLoss = new double[batchSize][1];
        double[] predictedVector, correctVector;
        for(int i = 0; i < batchSize; i++){
            predictedVector = predictions[i];
            correctVector = correct[i];
            totalLoss[i][0] = MatrixOps.l2Norm(MatrixOps.difference(predictedVector, correctVector));
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
