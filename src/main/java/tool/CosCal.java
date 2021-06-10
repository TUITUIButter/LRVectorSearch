package tool;

import org.nd4j.linalg.api.ndarray.INDArray;

public class CosCal {
    public static float CosCalculate(INDArray arg1, INDArray arg2){
        //计算余弦相似度
        float cos_sim = arg1.mmul(arg2.transpose()).getFloat(0,0)/
                (arg1.norm2Number().floatValue() * arg2.norm2Number().floatValue());
        return cos_sim;
    }

    public static float CosCalculate3(INDArray arg1, INDArray arg2){
        INDArray W1Enc = arg1.getRow(0);
        INDArray W2Enc = arg1.getRow(1);
        INDArray Q1Enc = arg1.getRow(2);
        INDArray Q2Enc = arg1.getRow(3);

        INDArray W3Enc = arg2.getRow(0);
        INDArray W4Enc = arg2.getRow(1);
        INDArray Q3Enc = arg2.getRow(2);
        INDArray Q4Enc = arg2.getRow(3);

        INDArray d1 = W1Enc.mmul(Q3Enc);
        INDArray d2 = W2Enc.mmul(Q4Enc);
        INDArray d = d1.add(d2);

        INDArray x1 = W1Enc.mmul(Q1Enc);
        INDArray x2 = W2Enc.mmul(Q2Enc);
        INDArray x3 = x1.add(x2);

        INDArray y1 = W3Enc.mmul(Q3Enc);
        INDArray y2 = W4Enc.mmul(Q4Enc);
        INDArray y3 = y1.add(y2);

        float x_q = (float) Math.pow(x3.getFloat(0,0),0.5);
        float y_q = (float) Math.pow(y3.getFloat(0,0),0.5);
        float f = d.getFloat(0,0)/(x_q * y_q);
        return f;
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
