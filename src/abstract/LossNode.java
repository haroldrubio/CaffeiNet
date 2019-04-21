import java.util.List;
public abstract class LossNode extends CompNode{
    protected double[][] predictions;
    protected double[][] batchLoss;
    public LossNode(List<Node> children, List<Node> parents){
        super(children, parents);
    }
    public void forward(double[][] predicted){
        predictions = predicted;
    }
    public void backward(double[][] correct){
        double[][] nextGradient = null;
        //Send the signal backwards
        if(correct != null) {
            batchLoss = this.f1.f(correct);
            nextGradient = this.derivatives1[0].deriv(correct);
        }
        super.passToParents(nextGradient);
    }

    protected void calculateLoss(double[][] correct){
        batchLoss = this.f1.f(correct);
    }

    public void printLoss(){
        if(batchLoss == null) return;
        int batchSize = batchLoss.length;
        double sumLoss = 0;
        for(int i = 0; i < batchSize; i++){
            sumLoss += Math.pow(batchLoss[i][0], 2);
        }
        sumLoss = Math.sqrt(sumLoss);
        System.out.printf("%5.3f\n", sumLoss);
    }

    public double getLoss(){
        if(batchLoss == null) return -1;
        int batchSize = batchLoss.length;
        double sumLoss = 0;
        for(int i = 0; i < batchSize; i++){
            sumLoss += Math.pow(batchLoss[i][0], 2);
        }
        sumLoss = Math.sqrt(sumLoss);
        return sumLoss;
    }

}
