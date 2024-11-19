from detectron2.data.datasets import register_coco_instances

def register_my_cityscapes():
    register_coco_instances("my_cityscapes_train", {}, \
        "/newdisk/suojinhui/object_detect/datasets/cityscapes_coco/annotations/instances_train2017_gai.json", \
            "/newdisk/suojinhui/object_detect/datasets/cityscapes_coco/train2017")
    register_coco_instances("my_cityscapes_val", {}, \
        "/newdisk/suojinhui/object_detect/datasets/cityscapes_coco/annotations/instances_val2017_gai.json", \
            "/newdisk/suojinhui/object_detect/datasets/cityscapes_coco/val2017")

    register_coco_instances("my_foggy_cityscapes_train", {}, \
        "/newdisk/suojinhui/object_detect/datasets/foggy_cityscapes_coco/annotations/instances_train2017.json", \
            "/newdisk/suojinhui/object_detect/datasets/foggy_cityscapes_coco/train2017")
    register_coco_instances("my_foggy_cityscapes_val", {}, \
        "/newdisk/suojinhui/object_detect/datasets/foggy_cityscapes_coco/annotations/instances_val2017.json", \
            "/newdisk/suojinhui/object_detect/datasets/foggy_cityscapes_coco/val2017")