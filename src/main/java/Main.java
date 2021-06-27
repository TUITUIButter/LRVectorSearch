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
import java.io.IOException;
import java.util.*;


public class Main {

    public static void main(String[] args) throws IOException {
        if(args.length == 0){
            System.err.println("需要在参数中指定配置文件");
            System.exit(-1);
        }

        String path = args[0];
        ParaBean para = ParaReader.ReadPara(path);
        String outPath = args[1];
        File file = new File(outPath);
        if(!file.exists()  && !file.isDirectory()){
            file.mkdir();
        }
        new Main().HandleWithOutPut(para,outPath);
    }

    void HandleWithOutPut(ParaBean para, String outPath) throws IOException {
        ReadData readData = new ReadData(
                para.wordVectorPath,
                para.docDictionaryPath,
                para.queryPath
        );

        FileWriter fw = null;
        BufferedWriter bfw = null;
        CSVPrinter csvPrinter = null;
        try{
            fw = new FileWriter(outPath + "/time.csv",true);
            csvPrinter = new CSVPrinter(fw, CSVFormat.DEFAULT);
            //bfw = new BufferedWriter(fw);
        }catch (Exception e){
            e.printStackTrace();
            System.exit(-1);
        }

        long startTime;   //获取开始时间
        long endTime; //获取结束时间
        Gson gson = new Gson();
        String paraInfo = gson.toJson(para);


        //bfw.write("\n");
        // 读取词向量
        readData.read_vector();
        System.out.println("开始构建倒排索引.....................");
        startTime = System.currentTimeMillis();
        HashMap<INDArray, HashMap<String, Double>> inverted_index = readData.read_keywords(para.keywordNum, para.docNumber);
        endTime = System.currentTimeMillis();
        System.out.println("结束构建倒排索引.....................");
        double docTime = endTime - startTime;
        //bfw.write("文档向量化时间： "+ (endTime - startTime) +"ms\n");

//        for(int k = 2; k <= 8;k++){
//            System.out.println("开始聚类.....................");
//            System.out.println("关键词集合大小: " + inverted_index.keySet().size());
//            startTime = System.currentTimeMillis();
//            ClusteringResults res = K_means.k_means(inverted_index.keySet(),-1,-1, 32*k, para.centerIt, para.threadNum);
//            endTime = System.currentTimeMillis();
//            System.out.println("结束聚类.....................");
//            double cluTime = endTime - startTime;
//            System.out.println((32*k) +": "+cluTime);
//            //bfw.write("聚类运行时间： "+ (endTime - startTime) +"ms\n");
//        }

        //聚类
        System.out.println("开始聚类............................");
        System.out.println("关键词集合大小: " + inverted_index.keySet().size());
        startTime = System.currentTimeMillis();
        ClusteringResults res = K_means.k_means(inverted_index.keySet(),para.centerNum, para.centerIt, para.threadNum);
        endTime = System.currentTimeMillis();
        System.out.println("结束聚类............................");
        double cluTime = endTime - startTime;
        //bfw.write("聚类运行时间： "+ (endTime - startTime) +"ms\n");

        Search search = new Search();

        System.out.println("开始读取查询........................");
        startTime = System.currentTimeMillis();
        File qPath = new File(para.queryPath);
        File[] tempList = qPath.listFiles();
        List<String> files = new ArrayList<String>();
        if (tempList == null) {
            System.err.println("file wrong!");
            System.exit(-1);
        }
        for (File value : tempList) {
            if (value.isFile()) {
                files.add(value.toString());
            }
            if (value.isDirectory()) {
                //这里就不递归了，
                String q = value.toString() + "\\desc.txt";
                files.add(q);
            }
        }

        for(String queries: files){
            String[] sp = queries.split("\\\\");
            String name = sp[sp.length - 2];
            ArrayList<INDArray> querys = readData.read_querys(queries);
            System.out.println(name+" 拓展..........................");
            ArrayList<INDArray> ex_clu = search.query_extend_two_cluster(querys,res, para.extendCenter, para.extendNum);
            System.out.println(name +" query拓展完毕....................");
            System.out.println(name +" 开始执行方案二检索.................");
            search.search_two(ex_clu, inverted_index, para.topK, outPath,name);
        }
        endTime = System.currentTimeMillis();
        System.out.println("读取查询结束.........................");
        double queryTime = endTime - startTime;

        // 方案二
        //不用聚类
//        startTime = System.nanoTime();
//        ArrayList<INDArray> extend_querys_two = search.query_extend_two(querys, inverted_index.keySet(), para.extendCenter * para.extendNum);
//        endTime = System.nanoTime();
//        double noCluExtTime = (endTime - startTime)/1000000.0;
        //bfw.write("查询拓展运行时间(无聚类)： "+ (endTime - startTime) +"ms\n");


//        startTime = System.nanoTime();
//        search.search_two(extend_querys_two, inverted_index, para.topK,outPath);
//        endTime = System.nanoTime();
//        double noCluSearchTime = (endTime - startTime)/1000000.0;
        //bfw.write("查询运行时间(无聚类)： "+ (endTime - startTime) +"ms\n");

        csvPrinter.printRecord(inverted_index.keySet().size(), para.centerNum,docTime,cluTime,queryTime);
        System.out.flush();
        //bfw.close();
        fw.close();
    }
}
