package perceptron;



import java.util.Random;

public class MLP {
    private double lrate = 1e-3 * 3;
    private double threshold = 1e-3;
    private int inputn;
    private int hiddenn = 240;
    private int outputn;
    private int layerSize = 4;
    private double momentum = 0.7;
    public double[] targetOut;
    public double[] error;
    public double betaSum;

    class alayer {
        int size;
        double out[];
        double deltaLow[];
        double deltaUp[][];
        double weight[][];
    };

    alayer layer[] = new alayer[layerSize];

    public MLP(int inputn, int outputn) {
        this.inputn = inputn;
        this.outputn = outputn;
        error = new double[outputn];
        // 레이어의 배열을 초기화
        layerSet();

        // 가중치에 -0.5 ~ 0.5 사이의 랜덤값을 입력
        Random r = new Random();
        double a;
        for (int n = 0; n < layerSize - 1; n++) {
            for (int i = 0; i < layer[n].size; i++) {
                for (int j = 0; j < layer[n + 1].size; j++) {
                    a = r.nextFloat() - 0.5f;
                    layer[n].weight[i][j] = a;
                }
            }
        }
    }

    private void layerSet() {
        // TODO Auto-generated method stub
        for (int i = 0; i < layerSize; i++) {
            layer[i] = new alayer();
        }
        layer[0].size = inputn;
        layer[1].size = hiddenn;
        layer[2].size = hiddenn;
        layer[3].size = outputn;

        for (int i = 0; i < layerSize; i++) {
            layer[i].out = new double[layer[i].size];
            if (i < layerSize - 1) {
                layer[i].weight = new double[layer[i].size][layer[i + 1].size];
                layer[i].deltaUp = new double[layer[i].size][layer[i + 1].size];
            }
            if (i != 0)
                layer[i].deltaLow = new double[layer[i].size];
        }

    }

    public void training(int[] input, int[] target) {

        targetOut = new double[outputn];
        // 받아온 input 값과 목표 출력 값을 저장
        for (int i = 0; i < input.length; i++) {
            layer[0].out[i] = input[i];
            if (i < 10) {
                targetOut[i] = target[i];
            }
        }
        double e = 0;
        // 출력 계산
        updateOutput();
        getError();
        e = errorSquare();
        System.out.println(e);
        // 오차 제곱의 합이 임계값보다 낮으면 역전파로 가중치를 변경하지 않음
        if (e > threshold)
            backPropagation();
    }

    private void backPropagation() {
        // TODO Auto-generated method stub
        getLowDelta();
        // 역전파 계산으로 가중치 업데이트
        for (int n = layerSize - 2; n >= 0; n--) {
            for (int i = 0; i < layer[n].size; i++) {
                for (int j = 0; j < layer[n + 1].size; j++) {
                    double delta = layer[n].deltaUp[i][j];
                    layer[n].deltaUp[i][j] = lrate * layer[n].out[i] * layer[n + 1].deltaLow[j];
                    layer[n].deltaUp[i][j] = momentum * delta + layer[n].deltaUp[i][j];
                    layer[n].weight[i][j] += layer[n].deltaUp[i][j];
                }
            }
        }
    }

    private void getLowDelta() {
        // TODO Auto-generated method stub
        // 소문자 델타를 미리 계산
        for (int k = 0; k < layer[layerSize - 1].size; k++) {
            layer[layerSize - 1].deltaLow[k] = layer[layerSize - 1].out[k]
                    * (1 - layer[layerSize - 1].out[k])* error[k];
        }
        for (int n = layerSize - 2; n > 0; n--) {

            for (int i = 0; i < layer[n].size; i++) {
                double sum = 0.0;
                for (int j = 0; j < layer[n + 1].size; j++) {
                    sum += layer[n + 1].deltaLow[j] * layer[n].weight[i][j];
                }
                layer[n].deltaLow[i] = layer[n].out[i] * (1 - layer[n].out[i]) * sum;
            }
        }
    }

    private void getError() {
        // TODO Auto-generated method stub

        // 오차를 계산해서 오차 배열에 저장
        error = new double[outputn];

        for (int i = 0; i < outputn; i++) {
            error[i] = targetOut[i] - layer[layerSize - 1].out[i];
        }
    }
    private void updateOutput() {
        // TODO Auto-generated method stub
        // 출력층의 출력을 계산
        for (int n = 0; n < layerSize - 1; n++) {
            for (int j = 0; j < layer[n + 1].size; j++) {
                double sum = 0.0;
                for (int i = 0; i < layer[n].size; i++) {
                    sum += layer[n].out[i] * layer[n].weight[i][j];
                }
                layer[n + 1].out[j] = sigmoid(sum);
            }

        }
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public double errorSquare() {
        // 오차 제곱을 계산
        double e = 0.0;
        for (int i = 0; i < error.length; i++) {
            e += error[i] * error[i];
        }
        // TODO Auto-generated method stub
        return e;
    }

    public boolean checkTarget(int[] input, int[] target) {
        targetOut = new double[outputn];
        // 입력으로 들어온 test 할 input과 목표 출력을 배열에 저장
        for (int i = 0; i < input.length; i++) {
            layer[0].out[i] = input[i];
            if (i < 10) {
                targetOut[i] = target[i];
            }
        }
        // 들어온 입력을 가지고 출력층의 출력을 계산
        updateOutput();
        int maxIndex = 0;

        // 출력층의 출력중 가장 큰 값을 1로 하고 나머지는 0으로 만든다.
        for (int i = 0; i < layer[layerSize - 1].size; i++) {
            if (layer[layerSize - 1].out[i] >= layer[layerSize - 1].out[maxIndex])
                maxIndex = i;

        }
        layer[layerSize - 1].out = new double[outputn];
        layer[layerSize - 1].out[maxIndex] = 1.0;

        for (int i = 0; i < layer[layerSize - 1].size; i++) {
            // 목표 출력과 출려층의 출력이 하나라도 다르면 false를 전송
            if (layer[layerSize - 1].out[i] != targetOut[i]) {
                return false;
            }
        }
        // 모두 같다면 true를 전송
        return true;

    }
}