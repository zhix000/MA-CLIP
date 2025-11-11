import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from plcc_srcc_cal import correlation_evaluation
import json

def main(image_paths, jsons, datasets):
    from Maclip import model
    scorer = model.Maclip()
    with torch.set_grad_enabled(False):
        for image_path, input_json in zip(image_paths, jsons):
            with open(input_json) as f:
                all_data = json.load(f)
                gts = [float(di["gt_score"]) for di in all_data]
            prs1 = []

            for llddata in tqdm(all_data):

                name = image_path + llddata["img_path"]

                predq0 = scorer(name, datasets) 
                predq0 = torch.mean(predq0)
                prs1.append(predq0.squeeze().cpu().numpy()) 

        plcc, srcc, rmse = correlation_evaluation(prs1, gts,is_plot=False, plot_path="")
        print("SRCC:{:.4f}, PLCC:{:.4f}, RMSE{:.4f}".format(srcc,plcc,rmse))



if __name__ == '__main__':
    image_paths_all = [
        "/data/cbl/IQA/datasets/live/", 
        "/data/cbl/IQA/zhicheng/AGRM/data/AGIQA-3K/",
        "/data/cbl/IQA/zhicheng/AGRM/data/AGIQA-1K/file/",
        '/data/cbl/IQA/datasets/CSIQ/dst_imgs_all/',
        '/data/cbl/IQA/datasets/TID2013/distorted_images/',
        "/data/cbl/IQA/datasets/kadid10k/images/",
        "/data/cbl/IQA/datasets/koniq-10k/1024x768/",
        '/data/cbl/IQA/datasets/SPAQ/SPAQ/Dataset/TestImage/'
    ]
    json_prefix = "./jsons/"
    dataset_config = {
        "livec": [json_prefix + "livec.json"],
        "AGIQA-3k": [json_prefix + "AGIQA-3k.json"],
        "AGIQA-1k": [json_prefix + "AGIQA-1k.json"],
        "CSIQ": [json_prefix + "csiq.json"],
        "TID2013": [json_prefix + "tid2013.json"],
        "kadid": [json_prefix + "kadid.json"],
        "koniq": [json_prefix + "koniq.json"],
        "SPAQ": [json_prefix + "spaq.json"],
    }

    for idx, (dataset_name, json_paths) in enumerate(dataset_config.items()):
        image_paths = [image_paths_all[idx]] 
        main(image_paths, json_paths, dataset_name)




