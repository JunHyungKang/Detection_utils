import os
import xmltodict
import json
import argparse


classes = ['kangaroo']


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='')
    opt = parser.parse_args()
    return opt


def run(opt):
    # Pascal VOC 데이터셋 경로
    voc_dir = opt.data_path

    # COCO JSON 파일 경로
    coco_file = os.path.join(os.path.dirname(opt.data_path), f"{os.path.basename(opt.data_pth)}.json")

    # COCO JSON 파일에 저장될 Annotation 정보
    coco_data = {
        "info": {"year": 2022},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    # 클래스 정보 추가
    for i, cls in enumerate(classes):
        coco_data["categories"].append({
            "id": i + 1,
            "name": cls,
            "supercategory": "object"
        })

    # 이미지 정보와 어노테이션 정보 추가
    image_id = 1
    annotation_id = 1
    for filename in os.listdir(voc_dir):
        if not filename.endswith(".xml"):
            continue

        with open(os.path.join(voc_dir, filename)) as f:
            xml_data = xmltodict.parse(f.read())

        # 이미지 정보 추가
        image_data = {
            "id": image_id,
            "file_name": xml_data["annotation"]["filename"],
            "height": int(xml_data["annotation"]["size"]["height"]),
            "width": int(xml_data["annotation"]["size"]["width"]),
            "license": 0
        }
        coco_data["images"].append(image_data)

        # 어노테이션 정보 추가
        if "object" not in xml_data["annotation"]:
            continue

        objects = xml_data["annotation"]["object"]
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            cls = obj["name"]
            bbox = obj["bndbox"]
            x, y, w, h = [int(bbox[k]) for k in
                          ["xmin", "ymin", "xmax", "ymax"]]
            annotation_data = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": classes.index(cls) + 1,
                "bbox": [x, y, w - x, h - y],
                "area": (w - x) * (h - y),
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation_data)
            annotation_id += 1

        image_id += 1

    # COCO JSON 파일 저장
    with open(coco_file, "w") as f:
        json.dump(coco_data, f)


if __name__ == '__main__':
    opt = parse_opt()
    run(opt)
