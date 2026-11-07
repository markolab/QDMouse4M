"""Dataset constants shared by preprocessing, training, inference, and evaluation."""

VIEWS = ["24383488", "24404505", "24404515", "24461855", "24514980", "24589937"]

KEYPOINTS = [
    "ear_L",
    "ear_R",
    "back_top",
    "back_middle",
    "back_bottom",
    "tail_base",
    "tail_middle",
    "tail_tip",
    "forepaw_L",
    "forepaw_R",
    "hindpaw_L",
    "hindpaw_R",
]

SKELETON_EDGES = [
    ("ear_L", "back_top"),
    ("ear_R", "back_top"),
    ("back_top", "back_middle"),
    ("back_middle", "back_bottom"),
    ("back_bottom", "tail_base"),
    ("tail_base", "tail_middle"),
    ("tail_middle", "tail_tip"),
    ("back_middle", "forepaw_L"),
    ("back_middle", "forepaw_R"),
    ("back_bottom", "hindpaw_L"),
    ("back_bottom", "hindpaw_R"),
]

TRAIN_SUBSETS_SMALL = [
    "train_0050",
    "train_0100",
    "train_0500",
    "train_1000",
    "train_5000",
    "train_10000",
]

TRAIN_SUBSETS_LARGE = [
    "train_15000_in_session_extra5k",
    "train_20000_in_session_extra10k",
    "train_15000_out_20251030",
    "train_15000_out_20251109_11",
    "train_20000_out_20251030_20251109_11",
]

LP3D_BACKBONES = {
    "dino": "vits_dino",
    "dinov2": "vits_dinov2",
    "dinov3": "vits_dinov3",
}

SLEAP_BACKBONES = ["unet", "convnext", "swint"]

