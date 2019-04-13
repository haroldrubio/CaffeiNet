import java.util.List;
public class WeightNode extends CompNode{
    private double[][] weights;
    private double learningRate;
    private ParamNode parameters;
    public WeightNode(List<Node> children, List<Node> parents, double learningRate){
        super(children, parents);
        this.learningRate = learningRate;
        this.f1 = new OneVarFunction() {
            @Override
            public double[][] f(double[][] input) {
                return MatrixOps.matrixMatrix(input, weights);
            }
        };

        this.derivatives1 = new OneVarDerivative[] {
                //[0] is derivative with respect to X
                new OneVarDerivative() {
                    @Override
                    public double[][] deriv(double[][] input) {
                        return MatrixOps.matrixMatrix(input, MatrixOps.transpose(weights));
                    }
                },
                //[1] is derivative with respect to W
                new OneVarDerivative() {
                    @Override
                    public double[][] deriv(double[][] input) {
                        return MatrixOps.matrixMatrix(MatrixOps.transpose(prevHidden), input);
                    }
                }
        };

    }

    public void forward(double[][] input){
        List<Node> parents = this.getParents();
        for(Node n: parents)
            if(n instanceof ParamNode){
                parameters = (ParamNode) n;
                weights = parameters.getCurrentParameters();
            }
        this.prevHidden = input;
        this.hiddenState = this.f1.f(input);
        CompNode nextComp;
        List<Node> currentChildren = this.getChildren();
        if(currentChildren != null)
            for(Node n: getChildren()){
                if(n instanceof CompNode){
                   nextComp = (CompNode) n;
                   nextComp.forward(hiddenState);
                }
            }
    }

    public void backward(double[][] loss){
        CompNode nextComp;
        List<Node> currentParents = this.getParents();
        double[][] nextGradient = null;
        if(loss == null)
            parameters.updateParameters(learningRate);
        else{
            int batchSize = loss.length;
            //First send the next update to the parameters
            double[][] nextUpdate = this.derivatives1[1].deriv(loss);
            parameters.passUpdate(nextUpdate, batchSize);
            //Then send the signal backwards
            nextGradient = this.derivatives1[0].deriv(loss);
        }
        if(currentParents != null)
            for(Node n: currentParents)
                if(n instanceof CompNode) {
                    nextComp = (CompNode) n;
                    nextComp.backward(nextGradient);
                }
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double[][] getHiddenState(){
        return this.hiddenState;
    }

    public double[][] getPrevHidden(){
        return this.prevHidden;
    }

    public void printParameters(){
        int numRows = weights.length, numCols = weights[0].length;
        System.out.println("--------------Weights---------------");
        for(int i = 0; i < numRows; i++){
            System.out.printf("Row %d: [", (i + 1));
            for(int j = 0; j < numCols; j++){
                System.out.printf("%3.2f, ", weights[i][j]);
            }
            System.out.print("}\n");
        }
        System.out.println("--------------Weights---------------");
    }

}
