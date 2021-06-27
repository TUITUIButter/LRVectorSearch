import com.google.gson.Gson;
import entity.ClusteringResults;
import entity.ParaBean;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.nd4j.linalg.api.ndarray.INDArray;
import tool.K_means;
import tool.ParaReader;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.HashMap;

public class TestKmeans
{
    public static void main(String[] args) {
        if(args.length == 0){
            System.err.println("需要在参数中指定配置文件");
            System.exit(-1);
        }

        String path = args[0];
        ParaBean para = ParaReader.ReadPara(path);

        ReadData readData = new ReadData(
                para.wordVectorPath,
                para.docDictionaryPath,
                para.queryPath
        );


        long startTime;   //获取开始时间
        long endTime; //获取结束时间
        Gson gson = new Gson();

        // 读取词向量
        readData.read_vector();
        System.out.println("开始构建倒排索引.....................");
        startTime = System.currentTimeMillis();
        HashMap<INDArray, HashMap<String, Double>> inverted_index = readData.read_keywords(para.keywordNum, para.docNumber);
        readData.SaveVec(10,10);
//        endTime = System.currentTimeMillis();
//        System.out.println("结束构建倒排索引.....................");
//        double docTime = endTime - startTime;
//
//        //聚类
//        System.out.println("开始聚类中心测试.....................");
//        System.out.println("关键词集合大小: " + inverted_index.keySet().size());
//        startTime = System.currentTimeMillis();
//        double[] res = K_means.k_cal(inverted_index.keySet(), para.threadNum);
//        endTime = System.currentTimeMillis();
//        System.out.println("结束聚类中心测试.....................");
//        System.out.println(Arrays.toString(res));
    }
}
