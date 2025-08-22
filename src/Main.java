import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations;

import java.io.*;
import java.util.ArrayList;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    final static int TRAIN_FILE_CHAR_LENGTH = 60000;

    public static void main(String[] args) throws IOException, ClassNotFoundException {
//        createNewWeightsAndBiases();


        ArrayList<SimpleMatrix> trainingInputData = new ArrayList<>();
        ArrayList<SimpleMatrix> expectedOutputs = new ArrayList<>();
        ArrayList<Integer> correctAnswers = new ArrayList<>();

        BufferedReader trainDataReader = new BufferedReader(new FileReader("src/mnist_train.txt"));
        for (int i = 0; i < TRAIN_FILE_CHAR_LENGTH; i++) {
            String in = trainDataReader.readLine();
            int correctAnswer = in.charAt(0)-48; //subtract 48 for char to int conversion
            correctAnswers.add(correctAnswer);
//            System.out.println("LENGTH OF TRAINING DATA STRING: " + ((float) in.length() -737));
            SimpleMatrix expectedOutput = new SimpleMatrix(10, 1);
            expectedOutput.fill(0);
            expectedOutput.set(in.charAt(0) - 48, 0, 1); //subtract 48 from read character to adjust for ASCII table
            expectedOutputs.add(expectedOutput);
            in = in.substring(2);

            trainingInputData.add(getTrainingInputMatrix(in));
        }

        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.readWB();
        for(int j = 0; j < 5; j++) {
            for(int i = 100; i >= 0; i--){
                ArrayList<SimpleMatrix> randTrainingData = new ArrayList<>();
                ArrayList<SimpleMatrix> randExpectedOutputs = new ArrayList<>();
                ArrayList<Integer> randCorrectAns = new ArrayList<>();
                for(int k = 0; k < NeuralNetwork.MINI_BATCH_SIZE; k++){ //randomizing training batches
                    int randIndex = (int) (Math.random() * (trainingInputData.size() - 1));
                    randTrainingData.add(trainingInputData.get(randIndex));
                    randExpectedOutputs.add(expectedOutputs.get(randIndex));
                    randCorrectAns.add(correctAnswers.get(randIndex));
                }
                neuralNetwork.backpropBatch(randTrainingData, randExpectedOutputs, randCorrectAns);
                neuralNetwork.adjustWB();
            }
        }
        neuralNetwork.writeWB();
    }

    private static SimpleMatrix getTrainingInputMatrix(String in) {
        SimpleMatrix trainCase = new SimpleMatrix(784, 1);
        int currPixelNum = 0;
        int j = -1;
        while(j < in.length() - 737){
            j++;
            int startIndex = j;
            while(in.charAt(j) != ',' && in.charAt(j) != '\n'){
                j++;
            }
            trainCase.set(currPixelNum, 0, Integer.parseInt(in.substring(startIndex, j)));
            currPixelNum++;
//                System.out.println("currPixelNum increase");
//                System.out.println("j: " + j);
        }
        return trainCase;
    }
    private static void createNewWeightsAndBiases() throws IOException {
        SimpleOperations.ElementOpReal expandToNegative = (int row, int col, double x) -> x*2-1;

        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream("src/weights"));
        ObjectOutputStream bout = new ObjectOutputStream(new FileOutputStream("src/biases"));

        ArrayList<SimpleMatrix> weights = new ArrayList<>(); //valid index from 1-3
        ArrayList<SimpleMatrix> biases = new ArrayList<>(); //valid index from 1-3

        //index 1 of array holds weights for layer 1, which are multiplied by activations of layer 0
        weights.add(SimpleMatrix.random(0, 0));
        biases.add(SimpleMatrix.random(0, 0));
        for(int i = 0; i < NeuralNetwork.LAYER_SIZES.length; i++){
            if(i != 0){
                biases.add(SimpleMatrix.random(NeuralNetwork.LAYER_SIZES[i], 1));
                biases.set(i, biases.get(i).elementOp(expandToNegative));
                weights.add(SimpleMatrix.random(NeuralNetwork.LAYER_SIZES[i], NeuralNetwork.LAYER_SIZES[i-1]));
                weights.set(i, weights.get(i).elementOp(expandToNegative));
            }
        }
        out.writeObject(weights);
        bout.writeObject(biases);
    }
}