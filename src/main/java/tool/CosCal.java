package tool;

import org.nd4j.linalg.api.ndarray.INDArray;

public class CosCal {
    public static float CosCalculate3(INDArray arg1, INDArray arg2){
        INDArray W1Enc = arg1.getRow(0);
        INDArray W2Enc = arg1.getRow(1);
        INDArray Q1Enc = arg1.getRow(2);
        INDArray Q2Enc = arg1.getRow(3);

        INDArray W3Enc = arg2.getRow(0);
        INDArray W4Enc = arg2.getRow(1);


        INDArray temp1 = W1Enc.sub(W3Enc);
        INDArray temp2 = W2Enc.sub(W4Enc);

        INDArray res = temp1.mmul(Q1Enc).add(temp2.mmul(Q2Enc));
        return res.getFloat(0,0);
    }

    public static float CosCalculate(INDArray arg1, INDArray arg2){
        float[][] f1 = arg1.toFloatMatrix();
        float[][] f2 = arg2.toFloatMatrix();
        float[] sub1 = new float[f1[0].length];
        float[] sub2 = new float[f2[0].length];
        for(int i = 0; i < f1[0].length;i++){
            sub1[i] = f1[0][i] - f2[0][i];
            sub2[i] = f1[1][i] - f2[1][i];
        }
        float sum1 = 0,sum2 = 0;
        for(int i = 0; i < f1[0].length;i++){
            sum1 +=  sub1[i] * f1[2][i];
            sum2 +=  sub2[i] * f1[3][i];
        }
        return sum1+sum2;
    }

    public static float CosCalculate2(INDArray arg1, INDArray arg2){
        float[][] f1 = arg1.toFloatMatrix();
        float[][] f2 = arg2.toFloatMatrix();

        float sum1 = 0,sum2 = 0,sum3 = 0,sum4 = 0;
        for(int i = 0; i < f1[0].length;i++){
            sum1 = sum1 + f1[0][i]*f2[2][i];
            sum2 = sum2 + f1[1][i]*f2[3][i];

            sum3 = sum3 + f1[0][i]*f1[2][i] + f1[1][i]*f1[3][i];
            sum4 = sum4 + f2[0][i]*f2[2][i] + f2[1][i]*f2[3][i];
        }
        float res = (sum1+sum2)/((float) Math.pow(sum3,0.5) * (float) Math.pow(sum4,0.5));
        return res;
    }
}
