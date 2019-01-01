package perceptron;

import java.util.Scanner;

public class TestingData {
    private int Epochs;
    private int testDataSize;
    private int inputSize;
    private int outputSize;
    private int[][] testingDataInput;
    private int[][] testingDataOutput;

    public TestingData(int testDataSize, int inputSize, int outputSize){
        this.testDataSize = testDataSize;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        testingDataInput = new int[testDataSize][inputSize];
        testingDataOutput = new int[testDataSize][outputSize];
    }

    public void insertTestingData(Scanner sc){
        for(int i=0;i<testDataSize;i++){
            int target = sc.nextInt();
            TargetOutTest(target, i);
            for (int j = 0; j < inputSize; j++) {
                if (sc.nextInt() > 0)
                    testingDataInput[i][j] = 1;
                else
                    testingDataInput[i][j] = 0;
            }
        }

    }

    public void TargetOutTest(int target, int n) {
        for (int i = 0; i < outputSize; i++) {
            if (i == target)
                testingDataOutput[n][i] = 1;
            else
                testingDataOutput[n][i] = 0;
        }

    }

    public void testing(MLP perceptron){
        int count=0;
        for (int i = 0; i < testDataSize; i++) {
            if (perceptron.checkTarget(testingDataInput[i], testingDataOutput[i])) {
                count++;
            }
        }
        System.out.println("최종 성공률 :  " + (double) count / (double) testDataSize);

    }
}
