import knn.KeyGen;
import knn.SecureKnn;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import tool.CosCal;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Test {
    public static void main(String[] args) throws InterruptedException {
        SecureKnn secureKnn  = new SecureKnn(new KeyGen(2,4).GenerateKey());

        float[] w1 = {10f,10f};
        float[] w2 = {10f,20f};
        float[] w3 = {100f,300f};
        float[] w4 = {10f,50f};

        INDArray w1e = Nd4j.create(w1,new int[]{1, w1.length});
        INDArray w2e = Nd4j.create(w2,new int[]{1, w1.length});
        INDArray w3e = Nd4j.create(w3,new int[]{1, w1.length});
        INDArray w4e = Nd4j.create(w4,new int[]{1, w1.length});

        INDArray encRes = secureKnn.EncWord(w1e);
        INDArray encRes2 = secureKnn.EncWord(w2e);
        INDArray encRes3 = secureKnn.EncWord(w3e);
        INDArray encRes4 = secureKnn.EncWord(w4e);

        INDArray temp1 = encRes.getRow(0).sub(encRes2.getRow(2));
        INDArray temp2 = encRes.getRow(1).sub(encRes2.getRow(3));
        float t1 = (float) Math.pow( temp1.norm2Number().floatValue(),2);
        float t2 = (float) Math.pow( temp2.norm2Number().floatValue(),2);
        float r12 = t1 + t2;

        temp1 = encRes.getRow(0).sub(encRes3.getRow(2));
        temp2 = encRes.getRow(1).sub(encRes3.getRow(3));
        t1 = (float) Math.pow( temp1.norm2Number().floatValue(),2);
        t2 = (float) Math.pow( temp2.norm2Number().floatValue(),2);
        float r13 = t1 + t2;

        temp1 = encRes.getRow(0).sub(encRes4.getRow(2));
        temp2 = encRes.getRow(1).sub(encRes4.getRow(3));
        t1 = (float) Math.pow( temp1.norm2Number().floatValue(),2);
        t2 = (float) Math.pow( temp2.norm2Number().floatValue(),2);
        float r14 = t1 + t2;

        float cos_12 = CosCal.CosCalculate3(encRes,encRes2);
        float cos_13 = CosCal.CosCalculate3(encRes,encRes3);
        float cos_14 = CosCal.CosCalculate3(encRes,encRes4);


        System.out.println("1-2: " + cos_12 + "\t\tr1-2: " + r12);
        System.out.println("1-3: " + cos_13 + "\t\tr1-3: " + r13);
        System.out.println("1-4: " + cos_14 + "\t\tr1-4: " + r14);

    }


}
