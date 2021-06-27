import entity.ClusteringResults;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import tool.CosCal;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.*;

public class Search {

    /**
     *  querys 方案一拓展
     * @param querys
     * @param keywords_set
     */
    public ArrayList<ArrayList<INDArray>> query_extend_one(ArrayList<INDArray> querys, Set<INDArray> keywords_set, int extend_num){
        // 矩阵运算求cos
        // querys_vec 一行为一个query
        // keywords_vec 一列为一个关键词
        // 将query和文档关键词构建向量
        INDArray querys_vec = Nd4j.vstack(querys);
        INDArray keywords_vec = Nd4j.vstack(keywords_set);
        // 求cos
        INDArray cos_sim = querys_vec.mmul(keywords_vec.transpose()).div
            (querys_vec.norm2(true, 1).mmul(keywords_vec.transpose().norm2(true, 0)));
        // 找每行的top-k
        ArrayList<ArrayList<INDArray>> extend_querys = new ArrayList<>();
        long row_lenth = cos_sim.shape()[0];
        long col_lenth = keywords_vec.shape()[1];
        for(long i = 0; i < row_lenth; ++i){
            ArrayList<INDArray> row_extend_querys = new ArrayList<>();
//            row_extend_querys.add(querys_vec.getRow(0).reshape(1, col_lenth)); // 包含原来的词
            int[] flat = argsort(cos_sim.getRow(i).toDoubleVector(), false);
            for(int j = 0; j < extend_num ; ++j){
                row_extend_querys.add(keywords_vec.getRow(flat[j]).reshape(1, col_lenth));
            }
            extend_querys.add(row_extend_querys);
        }
        return extend_querys;
    }

    /**
     * 第二种方案拓展
     * @param querys
     * @param keywords_set
     * @param extend_num
     * @return
     */
    public ArrayList<INDArray> query_extend_two(ArrayList<INDArray> querys, Set<INDArray> keywords_set, int extend_num){
        // 矩阵运算求cos
        // querys 一行为一个query
        // keywords_vec 一列为一个关键词
        // 将query和文档关键词构建向量
        System.out.println("开始执行方案二拓展...............");
        ArrayList<INDArray> extend_querys = new ArrayList<>();
        ArrayList<INDArray> keywords = new ArrayList<>(keywords_set);
        for (INDArray q : querys){
            double[] score = new double[keywords.size()];
            int i = 0;
            for (INDArray keyword:keywords){
                score[i] = CosCal.CosCalculate(q,keyword);
                i++;
            }
            int[] flat = argsort(score, true);
            for(int r = 0; r < extend_num; r++){
                extend_querys.add(keywords.get(flat[r]) );
            }
        }

/*        INDArray querys_vec = Nd4j.vstack(querys);
        INDArray keywords_vec = Nd4j.vstack(keywords_set);
        // 求cos
        System.out.println("开始计算cos...............");
        INDArray cos_sim = querys_vec.mmul(keywords_vec.transpose()).div
            (querys_vec.norm2(true, 1).mmul(keywords_vec.transpose().norm2(true, 0)));
        // 找每行的top-k
        long row_lenth = cos_sim.shape()[0];
        long col_lenth = keywords_vec.shape()[1];
        for(long i = 0; i < row_lenth; ++i){
//            row_extend_querys.add(querys_vec.getRow(0).reshape(1, col_lenth)); // 包含原来的词
            int[] flat = argsort(cos_sim.getRow(i).toDoubleVector(), false);
            for(int j = 0; j < extend_num ; ++j){
                extend_querys.add(keywords_vec.getRow(flat[j]).reshape(1, col_lenth));
            }
        }*/
        System.out.println("query拓展完毕cos...............");
        return extend_querys;
    }

    public ArrayList<INDArray> query_extend_two_cluster
            (ArrayList<INDArray> queries, ClusteringResults clu, int centerNum,int extendNum) {

        double[] score = new double[clu.k_meas.size()];
        int ind = 0;

        ArrayList<INDArray> extend_queries = new ArrayList<>();

        //扩展每个词
        for (INDArray q : queries){
            //遍历所有中心
            for (INDArray center : clu.k_meas) {
                score[ind] = CosCal.CosCalculate(center, q);
                ind++;
            }
            int[] flat = argsort(score,true); //flat就是倒序最近接近的中心下表
            for(int i = 0; i < centerNum;i++){
                for(int j = 0;j < extendNum;j++){
                    //如果这个类没有足够多元素，就跳过这个类
                    if(clu.setHashMap.get(flat[i]).size() <= j){
                        break;
                    }
                    //添加这个中心的第j个，中心内的倒叙已经排好
                    extend_queries.add(clu.setHashMap.get(flat[i]).get(j));

                }
            }
            ind = 0;
        }

        return extend_queries;
    }

    /**
     *  检索方案一 一个关键词一个关键词来排
     * @param extern_querys query
     * @param inverted_index
     */
    public void search_one(
        ArrayList<ArrayList<INDArray>> extern_querys,
        HashMap<INDArray, HashMap<String, Double>> inverted_index, int k){
        int query_size = extern_querys.size();
        Map<String, Double> rank_map = new HashMap<>();
        // 文档集合
        Set<String> doc_set = new HashSet<>();
        // 获取文档集合
        for(HashMap<String, Double> doc_map: inverted_index.values()){
            doc_set.addAll(doc_map.keySet());
        }
        // 构建所有文档集合
        for(String doc_tmp: doc_set){
            rank_map.put(doc_tmp, 0.0);
        }
        List<Map.Entry<String, Double>> list = null;  // 保存排好序的文档和分数
        System.out.println("开始执行方案一检索................");
        for(int i = query_size; i > 0; i--){
            int top_k = new Double(Math.pow(2,i-1)).intValue()*k;
            // 获取第i个词的拓展词
            for(INDArray query:extern_querys.get(query_size-i)){
                HashMap<String, Double> map = inverted_index.get(query);
                for(String doc: map.keySet()){
                    if (rank_map.keySet().contains(doc)) {
                        // rank_map中有的文档才加
                        rank_map.put(doc, rank_map.get(doc) + map.get(doc));
                    }
                    else{
                        // rank_map中没有的文档默认淘汰
                        continue;
                    }
                }
            }
            System.out.println("开始第"+(query_size-i+1)+"次排序.............");
            list = rank_doc(rank_map);
            if (top_k > list.size()){
                top_k = list.size();
            }
            rank_map = new HashMap<>();
            for (int j = 0; j < top_k;  j++) {
                rank_map.put(list.get(j).getKey(), list.get(j).getValue());
            }
        }
        if (k > list.size()){
            k = list.size();
        }
        System.out.println("top-k 文档排序结果：");
        for (int i = 0; i < k;  i++) {
            System.out.println(list.get(i).getKey()+":"+list.get(i).getValue());
        }
    }

    /**
     * 第二种方案
     * @param extern_querys
     * @param inverted_index
     * @param k
     */
    public void search_two(
        ArrayList<INDArray> extern_querys,
        HashMap<INDArray, HashMap<String, Double>> inverted_index, int k, String outPath,String name)
    {
        Map<String, Double> rank_map = new HashMap<>();
        // 开始检索
        for(int i = 0; i < extern_querys.size(); i++){
            HashMap<String, Double> map = inverted_index.get(extern_querys.get(i));
            if(map == null){
                System.out.println("None");
                System.out.println("i: " + i + "\t" + extern_querys.get(i).getDouble(0,0));
                continue;
            }
            for (String key: map.keySet()) {
                if(rank_map.containsKey(key)){
                    rank_map.put(key, rank_map.get(key) + map.get(key));
                }
                else{
                    rank_map.put(key, map.get(key));
                }

            }
        }
        // 排序
        List<Map.Entry<String, Double>> list = rank_doc(rank_map);
        if (k > list.size()){
            k = list.size();
        }
        System.out.println("k" +  k);
        File file = new File(outPath + "/res.txt");
        try {
            FileWriter fw = new FileWriter(file,true);
            BufferedWriter bfw = new BufferedWriter(fw);
            for (int i = 0; i < k;  i++) {
                bfw.write(name+" " + list.get(i).getKey()+" "+list.get(i).getValue());
                bfw.newLine();
            }
            bfw.write("\n\n");
            bfw.close();
            fw.close();
        }catch (Exception e){
            e.printStackTrace();
        }
        System.out.println("top-k 结果：已经写入"+ file.getAbsolutePath());
    }

    private List<Map.Entry<String, Double>> rank_doc(Map<String, Double> rank_map){
        // 排序
        List<Map.Entry<String, Double>> list = new ArrayList<>(rank_map.entrySet());

        Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
            //降序排序
            public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
                return o2.getValue().compareTo(o1.getValue());
            }
        });
        return list;
    }
    private static int[] argsort(final double[] a, final boolean ascending) {
        Integer[] indexes = new Integer[a.length];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {
            @Override
            public int compare(final Integer i1, final Integer i2) {
                return (ascending ? 1 : -1) * Double.compare(a[i1], a[i2]);
            }
        });
        return asArray(indexes);
    }
    private static <T extends Number> int[] asArray(final T... a) {
        int[] b = new int[a.length];
        for (int i = 0; i < b.length; i++) {
            b[i] = a[i].intValue();
        }
        return b;
    }
}
