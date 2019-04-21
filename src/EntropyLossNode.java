import java.util.List;

public class EntropyLossNode extends LossNode{
        public EntropyLossNode(List<Node> children, List<Node> parents){
            super(children, parents);
            this.f1 = new OneVarFunction() {
                @Override
                public double[][] f(double[][] input) {
                    return crossEntropy(input);
                }
            };
            this.derivatives1 = new OneVarDerivative[]{
                    new OneVarDerivative() {
                        @Override
                        public double[][] deriv(double[][] loss) {
                            return crossEntropyGrad(loss);
                        }
                    }
            };
        }
        private double[][] crossEntropy(double[][] correct){
            //Let the correct matrix be one-hot encodings of the classes
            int batchSize = correct.length, numClasses = correct[0].length;
            double[][] totalLoss = new double[batchSize][1];
            double[] predictedVector, correctVector;
            for(int i = 0; i < batchSize; i++){
                predictedVector = predictions[i];
                correctVector = correct[i];
                for(int k = 0; k < numClasses; k++)
                    totalLoss[i][0] += -1 * correctVector[k] * Math.log(predictedVector[k]);
            }
            return totalLoss;
        }

    /**
     * Returns the derivative of cross entropy loss through softmax
     * and into the hidden layer of the output node
     *
     * This gradient is provided by the CS231n page "Neural Networks Case Study"
     * @param correct A double matrix of the correct classes
     * @return Gradient with respect to the hidden layer
     */
    private double[][] crossEntropyGrad(double[][] correct){
            int batchSize = correct.length, numClasses = correct[0].length;
            double[][] gradient = new double[batchSize][numClasses];
            double[] predictedVector, correctVector;
            for(int i = 0; i < batchSize; i++){
                predictedVector = predictions[i];
                correctVector = correct[i];
                for(int k = 0; k < numClasses; k++)
                    gradient[i][k] +=  correctVector[k] * (predictedVector[k] - 1);
            }
            return gradient;
        }
}
