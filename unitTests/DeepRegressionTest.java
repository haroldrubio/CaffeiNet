import groovy.ui.Console;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Random;
import java.util.ArrayList;
public class DeepRegressionTest {
    int numTests = 1;

    @Test
    void singleLinearDim(){
        Random generateInput = new Random();
        for(int tests = 0; tests < numTests; tests++){
            int MAX_ITERS = 1000, MAX_VAL = 5;
            int inputDim = 1, outputDim = 1, hiddenDim = 3, batchSize = 16;
            double lr = 0.01, mu = 0.9, decay = 0.0005;
            NeuralNetwork net = new NeuralNetwork(lr, mu, decay);
            net.addLinearLayer(inputDim, hiddenDim);
            net.addLinearLayer(hiddenDim, hiddenDim);
            net.addRegressionLayer(hiddenDim, outputDim);

            double[][] inputData = new double[batchSize][inputDim];
            double[][] outputData = new double[batchSize][outputDim];
            int slope = generateInput.nextInt(10);
            int intercept = generateInput.nextInt(10);
            //-----------------------Train network-----------------------
            for(int iteration = 0; iteration < MAX_ITERS; iteration++){
                //-----------------------Get next batch-----------------------
                for(int batchNumber = 0; batchNumber < batchSize; batchNumber++){
                    for(int inDims = 0; inDims < inputDim; inDims++) {
                        inputData[batchNumber][inDims] = generateInput.nextDouble() * MAX_VAL;
                        outputData[batchNumber][inDims] = slope*inputData[batchNumber][inDims] + intercept;
                    }
                }
                //-----------------------Get next batch-----------------------
                net.forward(inputData);
                net.backpropagate(outputData);
                net.updateParameters();
                net.printLoss();
                //-----------------------Train network-----------------------

            }
        }
    }
}

