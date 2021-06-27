import knn.KeyGen;
import knn.SecureKnn;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import tool.CosCal;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Test {
    public static void main(String[] args) throws InterruptedException {
        long startTime;   //获取开始时间
        long endTime; //获取结束时间

        SecureKnn secureKnn  = new SecureKnn(new KeyGen(3,7).GenerateKey());

        float[] w1 = {1f,2f,3f};
        float[] w2 = {1f,5f,7f};
        float[] w3 = {1f,2f,7f};
        float[] w4 = {1f,2f,-1f};

        INDArray w1e = Nd4j.create(w1,new int[]{1, w1.length});
        INDArray w2e = Nd4j.create(w2,new int[]{1, w1.length});
        INDArray w3e = Nd4j.create(w3,new int[]{1, w1.length});
        INDArray w4e = Nd4j.create(w4,new int[]{1, w1.length});

        startTime = System.nanoTime();
        for( int i = 0; i < 2500;i++){
            INDArray encRes = secureKnn.EncWord(w1e);
            INDArray encRes2 = secureKnn.EncWord(w2e);
        }
        endTime = System.nanoTime();
        System.out.println((endTime - startTime)/1000000 +"ms");

//        INDArray encRes = secureKnn.EncWord(w1e);
//        INDArray encRes2 = secureKnn.EncWord(w2e);
//        INDArray encRes3 = secureKnn.EncWord(w3e);
//        INDArray encRes4 = secureKnn.EncWord(w4e);
//
//
//        startTime = System.nanoTime();
//        float cos_12 = CosCal.CosCalculate(encRes,encRes2);
//        float cos_13 = CosCal.CosCalculate(encRes,encRes3);
//        float cos_14 = CosCal.CosCalculate(encRes,encRes4);
//        endTime = System.nanoTime();
//        System.out.println((endTime - startTime) +"ns\n");
//
//        startTime = System.nanoTime();
//        float cos3_12 = CosCal.CosCalculate3(encRes,encRes2);
//        float cos3_13 = CosCal.CosCalculate3(encRes,encRes3);
//        float cos3_14 = CosCal.CosCalculate3(encRes,encRes4);
//        endTime = System.nanoTime();
//        System.out.println((endTime - startTime) +"ns\n");
//
//        System.out.println("1-2: " + cos_12 + "\t\tr1-2: " +cos3_12);
//        System.out.println("1-3: " + cos_13 + "\t\tr1-3: "+cos3_13);
//        System.out.println("1-4: " + cos_14 + "\t\tr1-4: "+cos3_14);
//
//        ArrayList<INDArray> arrayList = new ArrayList<>();
//
//        arrayList.add(encRes);
//        arrayList.add(encRes2);
//        arrayList.add(encRes3);
//        arrayList.add(encRes4);
//
//        float[][] s = CosCal.CosCalculateSet(arrayList);
//        System.out.println(Arrays.deepToString(s));

    }


}
