import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    final private float LEARNING_RATE = 1F;
    final static public int MINI_BATCH_SIZE = 256;
    final static public int[] LAYER_SIZES = {784, 16, 16, 10};

    private int batchesDone = 0;
    private int casesDone = 0;
    private int numCorrectAnswers = 0;


    ArrayList<SimpleMatrix> weights = new ArrayList<>(); //valid index from 1-3
    ArrayList<SimpleMatrix> biases = new ArrayList<>(); //valid index from 1-3
    ArrayList<SimpleMatrix> z = new ArrayList<>(); //valid index from 1-3
    ArrayList<SimpleMatrix> a = new ArrayList<>(); //0 to 3 index
    ArrayList<SimpleMatrix> deltaErr = new ArrayList<>(); //valid index from 1-3
    ArrayList<SimpleMatrix> nablaW = new ArrayList<>(4); //valid index from 1-3
    ArrayList<SimpleMatrix> nablaB = new ArrayList<>(4); //valid index from 1-3
    SimpleOperations.ElementOpReal sigmoid = (int row, int col, double value) -> 1.0/(1.0+Math.exp(-value));
    SimpleOperations.ElementOpReal sigmoidPrime = (int row, int col, double x) -> sigmoid(x)*(1-sigmoid(x));

    SimpleOperations.ElementOpReal ReLU = (int row, int col, double value) -> value <= 0 ? value*0.05 : value;
    SimpleOperations.ElementOpReal ReLUPrime = (int row, int col, double value) -> value <= 0 ? 0.05 : 1;

    public NeuralNetwork(){
        //index 1 of array holds weights for layer 1, which are multiplied by activations of layer 0
        weights.add(new SimpleMatrix(0, 0));
        biases.add(new SimpleMatrix(0, 0));
        z.add(new SimpleMatrix(0, 0));
        deltaErr.add((new SimpleMatrix(0, 0)));
        for(int i = 0; i < LAYER_SIZES.length; i++){
            a.add(new SimpleMatrix(1, LAYER_SIZES[i]));
            if(i != 0){
                z.add(new SimpleMatrix(LAYER_SIZES[i], 1));
                biases.add(new SimpleMatrix(LAYER_SIZES[i], 1));
                deltaErr.add(new SimpleMatrix(LAYER_SIZES[i], 1));
                weights.add(new SimpleMatrix(LAYER_SIZES[i], LAYER_SIZES[i-1]));
            }
        }
    }

    /**
     * feeds input data into the neural network and activates neurons layer by layer
     * @param input the 784 inputs of pixel brightness into the network
     */
    public void feedForward(SimpleMatrix input){
        a.set(0, input);
        for(int i = 1; i < a.size(); i++){
            z.set(i, weights.get(i).mult(a.get(i-1)));
            a.set(i, z.get(i).elementOp(sigmoid));
        }
//        z.set(3, weights.get(3).mult(a.get(3-1)));
//        a.set(3, z.get(3).elementOp(sigmoid));
    }
    private boolean isAnswerCorrect(int correctAnswer){
        SimpleMatrix out = a.get(3);
        float max = 0;
        int maxIndex = -1;
        for(int i = 0; i < 10; i++){
            if(out.getRow(i).get(0) > max){
                max = (float)out.getRow(i).get(0);
                maxIndex = i;
            }
        }
        System.out.print("Correct Answer: " + correctAnswer + " ");
        System.out.println("Network Output: " + maxIndex);
//        a.get(3).print();
//        weights.get(3).print();
        return maxIndex == correctAnswer;
    }

    /**
     * feedforwards and backpropogates through the network for one test case
     * @param expectedOutput the expected neuron outputs for last layer
     */
    private void backpropOneCase(SimpleMatrix input, SimpleMatrix expectedOutput){
        feedForward(input);
        deltaErr.set(3, (new SimpleMatrix(a.get(3).minus(expectedOutput))).elementMult(z.get(3).copy().elementOp(sigmoidPrime)));
        for(int i = 2; i > 0; i--){
            deltaErr.set(i, weights.get(i+1).transpose().mult(deltaErr.get(i+1)).elementMult(z.get(i).elementOp(sigmoidPrime)));
        }
        for(int i = 1; i < 4; i++){
            nablaB.set(i, nablaB.get(i).plus(deltaErr.get(i)));
            nablaW.set(i, nablaW.get(i).plus(deltaErr.get(i).mult(a.get(i-1).transpose())));
        }
    }

    /**
     * feedforwards and backpropogates through the network for one batch
     * @param inputs the array containing the 784 inputs of pixel brightness into the network for all cases in the batch
     * @param expectedOutputs the array containing all expected outputs for final neurons for all cases in batch
     * @param correctAnswers array containing all correct answers in single number format for each case in batch
     */
    public void backpropBatch(List<SimpleMatrix> inputs, List<SimpleMatrix> expectedOutputs, List<Integer> correctAnswers){
        nablaB = new ArrayList<>(4);
        nablaW = new ArrayList<>(4);
        for(int i = 0; i < 4; i++){
            nablaB.add(null);
            nablaW.add(null);
        }
        for(int i = 3; i > 0; i--){
            nablaB.set(i, new SimpleMatrix(biases.get(i).getNumRows(), 1));
            nablaW.set(i, new SimpleMatrix(weights.get(i).getNumRows(), weights.get(i).getNumCols()));
        }
        for(int i = 0; i < MINI_BATCH_SIZE; i++){
            backpropOneCase(inputs.get(i), expectedOutputs.get(i));
            casesDone++;
            if(isAnswerCorrect(correctAnswers.get(i))) numCorrectAnswers++;
            float accuracy = (float) numCorrectAnswers / casesDone;
            System.out.println("ACCURACY: " + accuracy * 100 + "%");
        }
        for(int i = 3; i > 0; i--){
            nablaB.set(i, nablaB.get(i).divide(MINI_BATCH_SIZE)); //divide to find average nablaB and nablaW
            nablaW.set(i, nablaW.get(i).divide(MINI_BATCH_SIZE));
        }
        batchesDone++;
        System.out.println("batchesDone: " + batchesDone);
    }

    /**
     * updates weights and biases based on nablaB and nablaW values calculated during backpropogation
     */
    public void adjustWB(){
        for(int i = 1; i < 4; i++){
            biases.set(i, biases.get(i).minus(nablaB.get(i).divide(1/LEARNING_RATE)));
            weights.set(i, weights.get(i).minus(nablaW.get(i).divide(1/LEARNING_RATE)));
        }
    }

    /**
     * saves the weights and biases to the local file
     */
    public void writeWB() throws IOException {
        ObjectOutputStream wOut = new ObjectOutputStream(new FileOutputStream("src/weights"));
        ObjectOutputStream bOut = new ObjectOutputStream(new FileOutputStream("src/biases"));
        wOut.writeObject(weights);
        bOut.writeObject(biases);
    }
    /**
     * reads weights and biases from the local file
     */
    public void readWB() throws IOException, ClassNotFoundException {
        ObjectInputStream wIn = new ObjectInputStream(new FileInputStream("src/weights"));
        ObjectInputStream bIn = new ObjectInputStream(new FileInputStream("src/biases"));
        weights = (ArrayList<SimpleMatrix>)wIn.readObject();
        biases = (ArrayList<SimpleMatrix>)bIn.readObject();
    }

    private double sigmoid(double x) {
        return 1.0/(1.0+Math.exp(-x));
    }



}
