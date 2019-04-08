import groovy.ui.Console;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Random;
import java.util.ArrayList;
public class DeepRegressionTest {
    int numTests = 100;
    @Test
    void singleDim(){
        double convergence = 0.01;
        double batchLoss = 0;
        for(int tests = 0; tests < numTests; tests++){
            int MAX_ITERS = 1000, MAX_VAL = 5;
            int inputDim = 1, outputDim = 1, hiddenDim = 3, batchSize = 10;
            double lr = 0.01;
            Random generateInput = new Random();
            //-----------------------Initialize network variables-----------------------
            double[][] hidden = MatrixOps.randomMatrix(inputDim + 1, hiddenDim);
            double[][] toOutput = MatrixOps.randomMatrix(hiddenDim, outputDim);
            InputNode inData = new InputNode(null);
            ParamNode hiddenParameters = new ParamNode(null, hidden);
            ParamNode outputParameters = new ParamNode(null, toOutput);
            WeightNode hiddenLayer = new WeightNode(null, null, lr);
            WeightNode outputLayer = new WeightNode(null, null, lr);
            //-----------------------Initialize network variables-----------------------

            //-----------------------Connect Computational Graph-----------------------
            ArrayList<Node> inChildren = new ArrayList<>(); inChildren.add(hiddenLayer);
            ArrayList<Node> hiddenParents = new ArrayList<>();
            hiddenParents.add(inData); hiddenParents.add(hiddenParameters);
            ArrayList<Node> hiddenChildren = new ArrayList<>(); hiddenChildren.add(outputLayer);
            ArrayList<Node> outputParents = new ArrayList<>();
            outputParents.add(hiddenLayer); outputParents.add(outputParameters);
            inData.setChildren(inChildren);
            hiddenLayer.setParents(hiddenParents); hiddenLayer.setChildren(hiddenChildren);
            outputLayer.setParents(outputParents);
            //-----------------------Connect Computational Graph-----------------------

            double[][] inputData = new double[batchSize][inputDim + 1];
            double[][] outputData = new double[batchSize][outputDim];
            double[] lossVector = new double[batchSize];
            double[] difference;
            double[][] predicted;
            double[][] lossMatrix = new double[batchSize][outputDim];
            //-----------------------Train network-----------------------
            for(int iteration = 0; iteration < MAX_ITERS; iteration++){
                //-----------------------Get next batch-----------------------
                for(int batchNumber = 0; batchNumber < batchSize; batchNumber++){
                    for(int inDims = 0; inDims < inputDim; inDims++) {
                        inputData[batchNumber][inDims] = generateInput.nextDouble() * MAX_VAL;
                        outputData[batchNumber][inDims] = 3*inputData[batchNumber][inDims] + 2;
                    }
                    inputData[batchNumber][inputDim] = 1;
                }
                //-----------------------Get next batch-----------------------
                inData.forward(inputData);
                predicted = outputLayer.getHiddenState();
                for(int i = 0; i < batchSize; i++){
                    difference = MatrixOps.difference(predicted[i], outputData[i]);
                    lossMatrix[i] = difference;
                    lossVector[i] = MatrixOps.l2Norm(difference);
                }
                batchLoss = MatrixOps.l2Norm(lossVector);
                //System.out.printf("Batch Loss: %5.4f\n", batchLoss);
                outputLayer.backward(lossMatrix);
                outputLayer.backward(null);
                //-----------------------Train network-----------------------

                //-----------------------Print parameters-----------------------
                //hiddenLayer.printParameters();
                //outputLayer.printParameters();
                //-----------------------Print parameters-----------------------

            }
            if(batchLoss >= convergence) System.err.printf("Loss not converged: %5.4f\n", batchLoss);
            assertTrue(batchLoss < convergence);
        }
    }
}