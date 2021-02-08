import sys
import logging
import os
import gensim

import jieba
from os.path import join

def load_stop_word(file_path):
    file_list = os.listdir(file_path)
    stop_word = dict()
    for i, file_name in enumerate(file_list):
        with open(join(file_path, file_name), "r", encoding="utf-8") as rf:
            for line in rf:
                words = line.strip().split(" ")
                for word in words:
                    stop_word[word] = 1
    return stop_word


def train_model(file_path, stop_word_file_dir, save_model_path):
    file_lists = os.listdir(file_path)
    documents = []
    stop_word_dict = load_stop_word(stop_word_file_dir)

    for i, file_name in enumerate(file_lists):
        if i < 10:
            print(file_name)
        words = cut_words_in_file(join(file_path, file_name), stop_word_dict)
        documents.append(gensim.models.doc2vec.TaggedDocument(words, file_name))
    model = gensim.models.Doc2Vec(documents, dm=1, size=100, window=8, workers=4)
    # model = gensim.models.Doc2Vec(documents, dm=1, vector_size=100, window=8, worker)
    model.save(save_model_path)


def cut_words_in_file(file_name, stop_word_dict):
    words = list()
    with open(file_name, "r", encoding="utf-8") as rf:
        for line in rf:
            seged = line.strip().split(" ")
            # seged = jieba.cut(line.strip(), HMM=True)
            for word in seged:
                if word not in stop_word_dict and word != "\t":
                    words.append(word)
    return words


def inference_vectors(model, target_doc, stop_word_dict):
    # model = gensim.models.doc2vec.Doc2Vec.load(model_path)
    words = cut_words_in_file(target_doc, stop_word_dict)
    return model.infer_vector(words, alpha=0.025, steps=500)


def inference_all_docs_vectors(model_path, file_dir, stop_word_file_dir, save_res_dir):
    # save_res_dir = "D:/_study_20_spring/IdeaProject/Doc2VecModel/" + "inference_res"
    model = gensim.models.doc2vec.Doc2Vec.load(model_path)
    stop_word_dict = load_stop_word(stop_word_file_dir)
    file_list = os.listdir(file_dir)
    file_name_to_vector_str = dict()
    file_name_to_similar_docs = dict()
    for i, file_name in enumerate(file_list):
        if i % 100 == 0:
            print(i, file_name)
        idx = file_name
        vector_str = inference_vectors(model, join(file_dir, file_name), stop_word_dict)
        sims = model.docvecs.most_similar([vector_str], topn=10)
        with open(os.path.join(save_res_dir, file_name), "w", encoding="utf-8") as wf:
            for sim_idx, sim in sims:
                wf.write(sim_idx + "\t" + str(sim) + "\n")
        wf.close()

    return file_name_to_vector_str, file_name_to_similar_docs


from sklearn.cluster import KMeans
import numpy as np

def load_file_vectors(model_path, file_dir) :
    # save_res_dir = "D:/_study_20_spring/IdeaProject/Doc2VecModel/" + "inference_res"
    model = gensim.models.doc2vec.Doc2Vec.load(model_path)
    stop_word_dict = load_stop_word(stop_word_file_dir)
    file_list = os.listdir(file_dir)
    idx_to_file_name = dict()
    idx_to_vector = dict()
    vectors = list()

    for i, file_name in enumerate(file_list):
        if i % 100 == 0:
            print(i, file_name)
        # if i == 1000:
        #     break
        idx = file_name
        vector_str = inference_vectors(model, join(file_dir, file_name), stop_word_dict)
        if i == 0:
            print(type(vector_str))
            print(vector_str.shape)
        idx_to_file_name[i] = file_name
        idx_to_vector[i] = vector_str
        vectors.append(vector_str.reshape((1, -1)))
    vectors = np.concatenate(vectors, axis=0)
    return idx_to_file_name, vectors


def cluster_k_means(vector_matrix, num_clusters, save_res_dir):
    # save_res_dir = "D:/_study_20_spring/IdeaProject/Doc2VecModel/" + "inference_res_vector"
    # num_clusters = 10
    km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40, init="k-means++", n_jobs=10)
    res = km_cluster.fit_predict(vector_matrix)
    np.save(join(save_res_dir, "kmeans_cluster_res_{:d}.npy".format(num_clusters)), res)


def from_numpy_to_cluster_dict(cluster_idx_res, idx_to_name, num_clusters, save_res_dir):
    num_items = cluster_idx_res.shape[0]
    cluster_idx_to_news_idx_array = dict()
    # save_res_dir = "D:/_study_20_spring/IdeaProject/Doc2VecModel/" + "inference_res_vector"
    print(num_items)
    for i in range(num_items):
        cluster_idx = int(cluster_idx_res[i])
        if cluster_idx not in cluster_idx_to_news_idx_array:
            cluster_idx_to_news_idx_array[cluster_idx] = [idx_to_name[i]]
        else:
            cluster_idx_to_news_idx_array[cluster_idx].append(idx_to_name[i])

    for i in cluster_idx_to_news_idx_array:
        idx_list = cluster_idx_to_news_idx_array[i]
        with open(join(save_res_dir, "k_means_cluster_res_num_clusters_{:d}_{:d}.txt".format(num_clusters, i)), "w") as wf:
            for idx in idx_list:
                wf.write(idx + "\n")
            wf.close()


def get_seg_text_for_each_news(fullfile_dir, seg_file_dir):
    stop_word_list = load_stop_word(stop_word_file_dir)
    for w in ['!',',','.','?','-s','-ly','</s>','s', "COVID-19", "Aug.", "GMT,", "GMT", "Aug", "(Xinhua)", "Xinhua", "--", "24", "2019"]:
        stop_word_list[w] = 1
    # fullfile_dir = "D:/_study_20_spring/IdeaProject/FullFile"
    # seg_file_dir = "D:/_study_20_spring/IdeaProject/SegFile"
    files = os.listdir(fullfile_dir)
    for i, file_name in enumerate(files):
        cut_word_list = cut_words_in_file(join(fullfile_dir, file_name), stop_word_list)
        with open(join(seg_file_dir, file_name), "w", encoding="utf-8") as wf:
            wf.write(" ".join(cut_word_list) + "\n")
        wf.close()

# todo: LDA for events cluster; 谱聚类？当我们有了文本向量之后？
# todo: tabLayout, on create animation... 时间的显示，觉得得到的本来就是news 所以可以获得具体的timeflag？
# todo: 完善一下逻辑？ 开始时聚类多少条新闻 调用update和more时获取得到的新闻如何处理等等
# TODO：LDA for events cluster 的后处理 --- 直接算出来每个要展示的events再显示吧 --- 探索一下不同的thread对应的新闻是怎样的 之后选一个合适的thread

if __name__ == "__main__":
    stop_word_file_dir = "D:/_study_20_spring/IdeaProject/stop_words_en"
    file_dir = "D:/_study_20_spring/IdeaProject/FullFile"
    save_model_path = "D:/_study_20_spring/IdeaProject/Doc2VecModel/doc2vec.model"
    # print("start training")
    # train_model(file_dir, stop_word_file_dir, save_model_path)
    # print("train over")
    # file_name_to_vector_str, file_name_to_similar_docs = inference_all_docs_vectors(save_model_path, file_dir, stop_word_file_dir)
    # print("inference over")

    save_res_dir = "D:/_study_20_spring/IdeaProject/Doc2VecModel/" + "inference_res_vector"

    # fullfile_dir = join("D:/_study_20_spring/IdeaProject", "ZhEvents")
    # seg_file_dir = join("D:/_study_20_spring/IdeaProject", "ZhSegEvents")

    fullfile_dir = join("D:/_study_20_spring/IdeaProject", "full_small_en_news")
    seg_file_dir = join("D:/_study_20_spring/IdeaProject", "seg_small_en_news")


    get_seg_text_for_each_news(fullfile_dir, seg_file_dir)

    save_model_path = join("D:/_study_20_spring/IdeaProject/Doc2VecModel", "doc2vec_zhEvents.model")

    print("start training")

    # train_model(fullfile_dir, stop_word_file_dir, save_model_path)

    print("end training")

    save_res_dir = join("D:/_study_20_spring/IdeaProject/Doc2VecModel", "zh_events_res")
    if not os.path.exists(save_res_dir):
        os.mkdir(save_res_dir)

    print("start inferencing")

    # inference_all_docs_vectors(save_model_path, fullfile_dir, stop_word_file_dir, save_res_dir)

    print("end inferencing")

    # idx_to_file_name, vectors = load_file_vectors(save_model_path, fullfile_dir)

    # np.save(join(save_res_dir, "file_vectors.npy"), vectors)
    # np.save(join(save_res_dir, "idx_to_file_name.npy"), idx_to_file_name)

    file_vectors = np.load(join(save_res_dir, "file_vectors.npy"))

    print("start k_means clustering")

    n_cluster = 7
    # cluster_k_means(file_vectors, num_clusters=n_cluster, save_res_dir=save_res_dir)


    # if not os.path.exists(save_res_dir):
    #     os.mkdir(save_res_dir)
    #
    # idx_to_file_name, vectors = load_file_vectors(save_model_path)
    # np.save(join(save_res_dir, "file_vectors.npy"), vectors)
    # np.save(join(save_res_dir, "idx_to_file_name.npy"), idx_to_file_name)

    # 143
    # print(type(idx_to_file_name))
    # print(idx_to_file_name[100])

    # file_vectors = np.load(join(save_res_dir, "file_vectors.npy"))

    # cluster_k_means(file_vectors)

    # cluster_idx_res = np.load(join(save_res_dir, "kmeans_cluster_res_{}.npy".format(str(n_cluster))))
    #
    # idx_to_file_name = np.load(join(save_res_dir, "idx_to_file_name.npy")).item()
    # #
    # from_numpy_to_cluster_dict(cluster_idx_res, idx_to_file_name, n_cluster, save_res_dir=save_res_dir)


    # num_clusters = 8
    # cluster_k_means(file_vectors, num_clusters)
    # cluster_idx_res = np.load(join(save_res_dir, "kmeans_cluster_res_{:d}.npy".format(num_clusters)))
    # from_numpy_to_cluster_dict(cluster_idx_res, idx_to_file_name, num_clusters)

    # get_seg_text_for_each_news()

    # if not os.path.exists(save_res_dir):
    #     os.mkdir(save_res_dir)
    #
    # with open(os.path.join(save_res_dir, "inferenced_vector.txt"), "w", encoding="utf-8") as wf:
    #     for file_name in file_name_to_vector_str:
    #         wf.write(file_name + "\t" + file_name_to_vector_str[file_name] + "\n")
    #     wf.close()
    #
    # print("saved 1")
    # for file_name in file_name_to_similar_docs:
    #     with open(os.path.join(save_res_dir, file_name), "w", encoding="utf-8") as wf:
    #         for similar_file_name in file_name_to_similar_docs[file_name]:
    #             wf.write(similar_file_name + "\t" + file_name_to_similar_docs[file_name][similar_file_name] + "\n")
    #     wf.close()
    # print("saved 2")
