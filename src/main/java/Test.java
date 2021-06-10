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
        SecureKnn secureKnn  = new SecureKnn(new KeyGen(3,7).GenerateKey());

        float[] w1 = {1f,2f,3f};
        float[] w2 = {0f,0f,0f};
        float[] w3 = {1f,2f,4f};
        float[] w4 = {2f,3f,4f};

        INDArray w1e = Nd4j.create(w1,new int[]{1, w1.length});
        INDArray w2e = Nd4j.create(w2,new int[]{1, w1.length});
        INDArray w3e = Nd4j.create(w3,new int[]{1, w1.length});
        INDArray w4e = Nd4j.create(w4,new int[]{1, w1.length});

        INDArray encRes = secureKnn.EncWord(w1e);
        INDArray encRes2 = secureKnn.EncWord(w2e);
        INDArray encRes3 = secureKnn.EncWord(w3e);
        INDArray encRes4 = secureKnn.EncWord(w4e);

        float cos_12 = CosCal.CosCalculate(encRes,encRes2);
        float cos_13 = CosCal.CosCalculate(encRes,encRes3);
        float cos_14 = CosCal.CosCalculate(encRes,encRes4);


        System.out.println("1-2: " + cos_12 + "\t\tr1-2: ");
        System.out.println("1-3: " + cos_13 + "\t\tr1-3: ");
        System.out.println("1-4: " + cos_14 + "\t\tr1-4: ");

    }


}
