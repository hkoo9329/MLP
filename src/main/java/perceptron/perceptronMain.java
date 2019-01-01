package perceptron;


import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
public class perceptronMain {
    private static String TraningDataName = "src/main/resources/MNIST.txt";
    private static int TrainingDataSize = 49000;
    private static int TestingDataSize = 21000;
    static int width = 28;
    static int height = 28;
    static int outputSize = 10;
    static int target;
    static int inputSize = width * height;
    private static int[][] input = new int[TrainingDataSize][inputSize];
    private static int[][] output = new int[TrainingDataSize][outputSize];


    public static void main(String[] args) {
        // TODO Auto-generated method stub
        MLP perceptron = new MLP(inputSize, outputSize);
        TrainingData trainingData = new TrainingData(TrainingDataSize, inputSize, outputSize);
        TestingData testingData = new TestingData(TestingDataSize, inputSize, outputSize);
        try {
            Scanner sc = new Scanner(new File(TraningDataName));
            System.out.println("input data scanning... ");

            // Training Data 와 Test Data를 저장
            sc = trainingData.insertTraningData(sc);
            testingData.insertTestingData(sc);
            trainingData.traning(perceptron);
            testingData.testing(perceptron);


        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }


}
