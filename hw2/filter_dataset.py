#!/usr/bin/env python3
import json
import os
import shutil
from pathlib import Path

from pycocotools.coco import COCO


CONFIG = {
    "coco_root": "./coco2017",  # корень полного COCO (annotations/, train2017/, val2017/)
    "split": "train",             # "train" или "val"
    "classes": [
        "cow", "car", "dog", "motorcycle", "bird",
        "sheep", "elephant", "horse", "train", "bench",
    ],
    "out_dir": "./mini_coco",   # куда положить mini-COCO
    "max_images_per_class": None,  # или число (int) для ограничения
    "ann_file": None,           # явный путь к annotations json (если None — берём из coco_root)
    "img_dir": None,            # явный путь к папке с картинками (если None — берём из coco_root)
}


def build_coco_subset(
    ann_file: str,
    img_dir: str,
    subset_cat_names,
    out_ann_file: str,
    out_img_dir: str,
    max_images_per_class=None,
):
    """
    Создаёт subset COCO:
      - копирует только нужные изображения в out_img_dir
      - пишет новый annotations.json (COCO формат) с фильтрованными категориями/аннотациями.
    """
    os.makedirs(out_img_dir, exist_ok=True)

    coco = COCO(ann_file)

    # Выбираем категории по именам
    cats = coco.loadCats(coco.getCatIds())
    name_to_id = {c["name"]: c["id"] for c in cats}
    missing = [n for n in subset_cat_names if n not in name_to_id]
    if missing:
        raise ValueError(f"Categories not found in COCO: {missing}")

    allowed_cat_ids = [name_to_id[n] for n in subset_cat_names]
    print(f"Allowed categories ({len(allowed_cat_ids)}): {subset_cat_names}")

    # Собираем список image_id, опционально ограничивая кол-во на класс
    img_ids_set = set()
    if max_images_per_class is None:
        for cid in allowed_cat_ids:
            ids = coco.getImgIds(catIds=[cid])
            img_ids_set.update(ids)
    else:
        for cid in allowed_cat_ids:
            ids = coco.getImgIds(catIds=[cid])
            ids = ids[:max_images_per_class]
            img_ids_set.update(ids)
    img_ids = sorted(list(img_ids_set))

    print(f"Selected {len(img_ids)} images for subset")

    images = coco.loadImgs(img_ids)

    # Фильтруем аннотации по выбранным категориям и картинкам
    ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=allowed_cat_ids, iscrowd=None)
    annotations = coco.loadAnns(ann_ids)
    print(f"Selected {len(annotations)} annotations for subset")

    # Оставляем только нужные категории
    categories = coco.loadCats(allowed_cat_ids)

    # Пересобираем ID (компактные последовательные id начиная с 1)
    old_img_id_to_new = {img["id"]: i + 1 for i, img in enumerate(images)}
    for i, img in enumerate(images):
        img["id"] = i + 1

    for i, ann in enumerate(annotations):
        ann["id"] = i + 1
        ann["image_id"] = old_img_id_to_new[ann["image_id"]]

    # Переписываем category_id -> компактные [1..K]
    old_cat_id_to_new = {cid: i + 1 for i, cid in enumerate(allowed_cat_ids)}
    for i, cat in enumerate(categories):
        cat["id"] = i + 1
    for ann in annotations:
        ann["category_id"] = old_cat_id_to_new[ann["category_id"]]

    # Копируем изображения
    for img in images:
        src_path = os.path.join(img_dir, img["file_name"])
        dst_path = os.path.join(out_img_dir, img["file_name"])
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Image not found: {src_path}")
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

    # Собираем итоговый COCO json
    coco_subset = {
        "info": coco.dataset.get("info", {}),
        "licenses": coco.dataset.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(out_ann_file, "w") as f:
        json.dump(coco_subset, f)
    print(
        f"Saved subset: {len(images)} images, {len(annotations)} anns -> {out_ann_file}"
    )


def main():
    coco_root = Path(CONFIG["coco_root"])
    out_root = Path(CONFIG["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    split = CONFIG["split"]
    split_name = "train2017" if split == "train" else "val2017"

    ann_file = (
        Path(CONFIG["ann_file"])
        if CONFIG["ann_file"] is not None
        else coco_root / "annotations" / f"instances_{split_name}.json"
    )
    img_dir = (
        Path(CONFIG["img_dir"])
        if CONFIG["img_dir"] is not None
        else coco_root / split_name
    )

    out_img_dir = out_root / split_name
    out_ann_dir = out_root / "annotations"
    out_ann_dir.mkdir(exist_ok=True, parents=True)
    out_ann_file = out_ann_dir / f"instances_{split_name}_subset.json"

    print("COCO root:", coco_root)
    print("Split:", split)
    print("Input ann:", ann_file)
    print("Input img dir:", img_dir)
    print("Output ann:", out_ann_file)
    print("Output img dir:", out_img_dir)
    print("Classes:", CONFIG["classes"])
    print("Max images per class:", CONFIG["max_images_per_class"])

    build_coco_subset(
        ann_file=str(ann_file),
        img_dir=str(img_dir),
        subset_cat_names=CONFIG["classes"],
        out_ann_file=str(out_ann_file),
        out_img_dir=str(out_img_dir),
        max_images_per_class=CONFIG["max_images_per_class"],
    )


if __name__ == "__main__":
    main()
