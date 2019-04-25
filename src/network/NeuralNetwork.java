
import java.util.ArrayList;
import java.util.List;
public class NeuralNetwork {
    private InputNode input;
    private Node lastLayer;
    private List<Node> layers;
    private double learningRate, momentum, decay;
    public NeuralNetwork(double learningRate, double momentum, double decay){
        List<Node> inputChildren = new ArrayList<>();
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.decay = decay;
        layers = new ArrayList<>();
        input = new InputNode(inputChildren);
        lastLayer = input;
        layers.add(input);
    }

    public void addLinearLayer(int inputDims, int outputDims){
        CompNode linearLayer = new LinearLayerTan(null, null, inputDims, outputDims,
                                                  this.learningRate, this.momentum, this.decay);
        this.addLayer(linearLayer);
    }

    public void addClassifierLayer(int hiddenDims, int outputDims){
        CompNode linearLayer = new ClassifierLayer(null, null, hiddenDims, outputDims,
                this.learningRate, this.momentum, this.decay);
        this.addLayer(linearLayer);
    }

    public void addRegressionLayer(int hiddenDims, int outputDims){
        CompNode regressionLayer = new RegressionLayer(null, null, hiddenDims, outputDims,
                this.learningRate, this.momentum, this.decay);
        this.addLayer(regressionLayer);
    }

    public void forward(double[][] input){
        this.input.forward(input);
    }

    public void backpropagate(double[][] correct){
        this.updateHyperparameters();
        CompNode lastComputation = (CompNode) lastLayer;
        lastComputation.backward(correct);
    }

    public void updateParameters(){
        this.backpropagate(null);
    }

    private void addLayer(CompNode nextLayer){
        lastLayer.getChildren().add(nextLayer);
        nextLayer.getParents().add(lastLayer);
        layers.add(nextLayer);
        lastLayer = nextLayer;
    }

    private void updateHyperparameters(){
        LinearLayer currentLinear;
        OutputLayer currentOutput;
        for(Node n : layers){
            if(n instanceof LinearLayer) {
                currentLinear = (LinearLayer) n;
                currentLinear.setDecay(this.decay);
                currentLinear.setMomentum(this.momentum);
                currentLinear.setLearningRate(this.learningRate);
            }
            else if (n instanceof OutputLayer){
                currentOutput = (OutputLayer) n;
                currentOutput.setDecay(this.decay);
                currentOutput.setMomentum(this.momentum);
                currentOutput.setLearningRate(this.learningRate);
            }
        }
    }

    /**
     * Sets this network's learning rate
     * @param learningRate A double
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Sets this network's momentum coefficient
     * @param momentum A double
     */
    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    /**
     * Set this network's weight decay coefficient
     * @param decay A double
     */
    public void setDecay(double decay) {
        this.decay = decay;
    }

    public void printLoss(){
        OutputLayer last = (OutputLayer) lastLayer;
        last.printLoss();
    }

    public void calculateLoss(double[][] correct){
        OutputLayer last = (OutputLayer) lastLayer;
        last.calculateLoss(correct);
    }

    public double getLoss(){
        OutputLayer last = (OutputLayer) lastLayer;
        return last.getLoss();
    }

    public double getAccuracy(double[][] correct){
        ClassifierLayer last = (ClassifierLayer) lastLayer;
        return last.getAccuracy(correct);
    }

    public double[][] getPredictions(){
        OutputLayer last = (OutputLayer) lastLayer;
        return last.getPredictions();
    }

}
