package perceptron;

import lombok.Data;

import java.util.Scanner;

@Data
public class TrainingData {
    private int Epochs = 50;
    private int TrainingDataSize;
    private int outputSize;
    private int inputSize;
    private int[][] TrainingDataInput;
    private int[][] TrainingDataOutput;

    public TrainingData(int TraningDataSize, int inputSize, int outputSize) {
        this.TrainingDataSize = TraningDataSize;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.TrainingDataInput = new int[TrainingDataSize][inputSize];
        this.TrainingDataOutput = new int[TrainingDataSize][outputSize];
    }

    public Scanner insertTraningData(Scanner sc) {
        for (int i = 0; i < TrainingDataSize; i++) {
            int target = sc.nextInt();
            TargetOut(target, i);
            for (int j = 0; j < inputSize; j++) {
                // 데이터의 0보다 큰값은 1로 변환해서 저장
                // 처리속도 향상을 위해
                if (sc.nextInt() > 0)
                    TrainingDataInput[i][j] = 1;
                else
                    TrainingDataInput[i][j] = 0;
            }
        }
        return sc;
    }

    // 목표 출력을 배열로 변환
    public void TargetOut(int target, int n) {
        for (int i = 0; i < outputSize; i++) {
            if (i == target)
                TrainingDataOutput[n][i] = 1;
            else
                TrainingDataOutput[n][i] = 0;
        }

    }

    public void traning(MLP perceptron){
        int count=0;
        int check = 0;
        double percent = 0;
        loop:
        for (int i = 0; i < Epochs; i++) {
            for (int n = 0; n < TrainingDataSize; n++) {
                // MLP에서 학습을 시작
                perceptron.training(TrainingDataInput[n], TrainingDataOutput[n]);
                if (perceptron.errorSquare() < 1e-3)
                    check++;
                // Training Data의 오차제곱값이 모두 임계값 이내라면 반복문 종료
                if (check == TrainingDataSize - 1)
                    break loop;
                // Training Data에서의 성공횟수를 계산
                if (perceptron.checkTarget(TrainingDataInput[n], TrainingDataOutput[n])) {
                    count++;
                }
            }
            percent = (double) count / TrainingDataSize;
            System.out.println("Epochs " + i + " :  " + percent);
            check = 0;
            count = 0;
            // Training Data의 성공률이 98.5%이상이라면 Test Data의 성공률의 변화는 미미함으로
            // 반복문을 종료
            if (percent > 0.985)
                break;
        }
    }


}
