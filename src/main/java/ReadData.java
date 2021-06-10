import entity.Key;
import knn.KeyGen;
import knn.SecureKnn;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;


public class ReadData {
    private final String m_vector_file;     //词向量文件
    private final String m_querys_file;     //查询文件
    private final String m_keywords_file;   //关键词文件
    private Word2Vec m_word2Vec;      // 词向量文件
    Key key; //knn加密密钥

    public ReadData(String vector_file, String keywords_file,String querys_file) {
        m_vector_file = vector_file;
        m_querys_file = querys_file;
        m_keywords_file = keywords_file;
        key = new KeyGen(300,302).GenerateKey();
    }

    /**
     * 读取词向量
     */
    public void read_vector() {
        System.out.println("开始读取词向量..............................");
        File gModel = new File(m_vector_file);
        m_word2Vec = WordVectorSerializer.readWord2VecModel(gModel, true);
        System.out.println("读取词向量完成................");
    }

    /**
     * 读取预处理后的查询词
     *
     * @return String[]
     */
    public ArrayList<INDArray> read_querys() {
        String encoding = "UTF-8";
        File file = new File(m_querys_file);
        Long filelength = file.length();
        byte[] filecontent = new byte[filelength.intValue()];
        try {
            FileInputStream in = new FileInputStream(file);
            in.read(filecontent);
            in.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            return query2vector(new String(filecontent, encoding).split("\\s+"));

        } catch (UnsupportedEncodingException e) {
            System.err.println("The OS does not support " + encoding);
            e.printStackTrace();
            return null;
        }
    }

    /**
     * 读取 文档关键词，构建倒排索引
     *
     * @param keywords_num 读取关键词数量
     * @return HashMap<INDArray, HashMap < String, Double>>
     * Hashmap<关键词的词向量, hashmap<文档名，单词在文档的相关分数>>
     */

    public HashMap<INDArray, HashMap<String, Double>> read_keywords(int keywords_num,int document_num) {
        HashMap<INDArray, HashMap<String, Double>> inverted_index = new HashMap<>();
        List<String> files = new ArrayList<String>();
        File file = new File(m_keywords_file);
        File[] tempList = file.listFiles();

        //生成加密util
        SecureKnn secureKnn = new SecureKnn(key);

        if(tempList == null){
            System.err.println("keywords file wrong!");
            System.exit(-1);
        }
        for (File value : tempList) {
            if (value.isFile()) {
                files.add(value.toString());
            }
            if (value.isDirectory()) {
                //这里就不递归了，
                File tempFile = new File(value.toString());
                File[] subject = tempFile.listFiles();
                assert subject != null;
                for (File doc : subject) {
                    files.add(doc.toString());
                }
            }
        }

        int doc_cnt = 0;
        READDOCS:
        for (String doc : files) {
            try {
                FileReader fr = new FileReader(doc);
                BufferedReader bf = new BufferedReader(fr);
                String str;
                // 按行读取字符串
                int flag = 0;
                String doc_name = "";
                while ((str = bf.readLine()) != null) {
                    String[] str_array = str.split("\\s+");
                    if (str_array.length == 1) {
                        /* 文档名 */
                        doc_name = str;
                        doc_cnt++;
                        //如果读取的文档数量达到了目标，则返回
                        if(doc_cnt > document_num){
                            break READDOCS;
                        }
                        flag = 0;
                    } else {
                        /* 这个第一个是分数， 第二个是keyword */
                        if (flag < keywords_num) {
                            // 关键词词向量化
                            INDArray keyword_vec = m_word2Vec.getWordVectorMatrix(str_array[1]);
                            keyword_vec = secureKnn.EncWord(keyword_vec);
                            Set<INDArray> keywords_set = inverted_index.keySet();
                            if (keywords_set.contains(keyword_vec)) {
                                inverted_index.get(keyword_vec).put(doc_name, Double.valueOf(str_array[0]));
                            } else {
                                HashMap<String, Double> docs_map = new HashMap<>();
                                docs_map.put(doc_name, Double.valueOf(str_array[0]));
                                inverted_index.put(keyword_vec, docs_map);
                            }
                            ++flag;
                        }
                    }
                }
                System.out.println("文档数量：" + doc_cnt);
                bf.close();
                fr.close();
            } catch (IOException e) {
                e.printStackTrace();
                return null;
            }

        }
        return inverted_index;
    }


    /**
     * query词转变为词向量
     *
     * @return
     */

    private ArrayList<INDArray> query2vector(String[] query_str) {
        SecureKnn secureKnn = new SecureKnn(key);
        ArrayList<INDArray> query_vec = new ArrayList<>();
        for (String query : query_str) {
            INDArray temp = m_word2Vec.getWordVectorMatrix(query);
            temp = secureKnn.EncWord(temp);
            query_vec.add(temp);
        }
        return query_vec;
    }

//1


}
