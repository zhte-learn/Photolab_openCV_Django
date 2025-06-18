"""
similarity_retrieval.py

Image‐similarity retrieval application using OpenCV.
Now displays per‐metric scores and the aggregation formula.
"""

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# config
# sets up and runs a multi-stage image similarity retrieval,
# comparing a hard-coded input image against a pool
# and returning the top three matches.
STANDARD_SIZE    = (300, 300)
IMAGE_POOL_DIR   = "/content/image_pool"     # directory of pool images

# defines the folder used to save and load pickled
# feature dictionaries, avoiding recomputation on subsequent runs
FEATURE_CACHE    = "/content/feature_cache"
INPUT_IMAGE_PATH = "/content/proj_test6.png"
TOP_K            = 3

os.makedirs(FEATURE_CACHE, exist_ok=True)

# reads the image from disk, resizes it to standard size,
# and then computes and returns a dictionary
def extract_all_features(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, STANDARD_SIZE)

    # Color histogram (HSV)
    # convert the resized BGR image to HSV,
    # compute an 8×8×8‐bin 3D histogram over
    # the H, S, and V channels, and normalize it
    # so that the histogram sums to one

    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    cv2.normalize(hist, hist, cv2.NORM_L1)

    # Grayscale
    # convert the resized image to a single–channel grayscale
    # image and make a copy of that grayscale as the “template” image
    # used later for template matching

    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = gray.copy()

    # ORB descriptors
    # initialize an ORB detector with up to 500 features
    # and then detect keypoints and compute their
    # binary descriptors on the grayscale image

    orb       = cv2.ORB_create(nfeatures=500)
    _, des    = orb.detectAndCompute(gray, None)

    # Canny edges
    # applies a Canny edge detector (with thresholds 100 and 200)
    # to the grayscale image, producing a binary edge map

    edges     = cv2.Canny(gray, 100, 200)

    # Contours
    # threshold the grayscale image into black-and-white,
    # then find all external contours in that binary mask

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return {
        "path":      image_path,
        "hist":      hist.flatten(),
        "gray":      gray,
        "template":  template,
        "orb_des":   des,
        "edges":     edges,
        "contours":  contours
    }

# checks for a cached pickle file named <basename>.pkl in FEATURE_CACHE;
# if present, it unpickles and returns the stored feature dict,
# otherwise calls extract_all_features(...), pickles its result for future reuse,
# and then returns that newly computed feature dict.

def load_or_compute_features(image_path: str) -> dict:
    base       = os.path.splitext(os.path.basename(image_path))[0]
    cache_file = os.path.join(FEATURE_CACHE, f"{base}.pkl")
    if os.path.exists(cache_file):
        return pickle.load(open(cache_file,"rb"))
    feats = extract_all_features(image_path)
    with open(cache_file, "wb") as f:
        pickle.dump(feats, f)
    return feats

# Smilarity metrics
# defines a suite of functions that each compute a normalized
# similarity score between two images—based on color histograms,
# structural similarity, ORB feature matches, template matching,
# edge overlap, and contour shape—and then aggregates those
# six scores into a single average similarity

# computes the correlation between two HSV color histograms
# (after converting them to float32), yielding a value in [−1, 1]
# that measures how closely their color distributions match
def compare_hist(h1, h2):
    return float(cv2.compareHist(h1.astype('float32'), h2.astype('float32'), cv2.HISTCMP_CORREL))

# calculates the Structural Similarity Index (SSIM)
# on two grayscale images,returning a float in [0, 1]
# that quantifies their perceptual similarity in terms of
# luminance, contrast, and structure
def compare_ssim(g1, g2): # huh?
    score, _ = ssim(g1, g2, full=True)
    return float(score)
# matches two sets of ORB descriptors with a Hamming‐distance
# matcher and returns the ratio of matched keypoints to the larger
# descriptor count, giving a normalized feature‐matching score in [0, 1]
def compare_orb(d1, d2): # huh?
    if d1 is None or d2 is None:
        return 0.0
    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    return len(matches) / max(len(d1), len(d2), 1)

# runs normalized cross‐correlation template matching of one
# grayscale patch against another and returns the maximum response
# as a float in [−1, 1], indicating the best alignment.
def compare_template(t1, t2): # huh?
    res = cv2.matchTemplate(t1, t2, cv2.TM_CCOEFF_NORMED)
    return float(np.max(res))

# treats each Canny edge map as a binary mask and computes
# their Jaccard index (intersection over union), yielding
# a similarity ratio in [0, 1] of overlapping edge pixels.

def compare_edges(e1, e2): # huh?
    inter = np.logical_and(e1>0, e2>0).sum()
    union = np.logical_or(e1>0, e2>0).sum()
    return float(inter/union) if union>0 else 0.0

# uses OpenCV’s matchShapes on the first contour of each image,
# converts the resulting shape‐distance into
# a similarity score via 1/(1+dist), and returns that float in (0, 1].
def compare_contours(c1, c2): # huh?
    if not c1 or not c2:
        return 0.0
    dist = cv2.matchShapes(c1[0], c2[0], cv2.CONTOURS_MATCH_I1, 0.0)
    return float(1.0/(1.0+dist))

# invokes all six of the above metrics on the corresponding feature
# arrays/dicts, collects them into a dictionary, computes their
# arithmetic mean, and returns both the overall average and
# the per‐metric breakdown.
def compare_features(f1: dict, f2: dict): # huh?
    scores = {
        "hist":     compare_hist(f1["hist"],    f2["hist"]),
        "ssim":     compare_ssim(f1["gray"],    f2["gray"]),
        "orb":      compare_orb(f1["orb_des"],  f2["orb_des"]),
        "templ":    compare_template(f1["template"], f2["template"]),
        "edges":    compare_edges(f1["edges"],   f2["edges"]),
        "contours": compare_contours(f1["contours"], f2["contours"])
    }
    avg = float(np.mean(list(scores.values())))
    return avg, scores


# takes the precomputed features of a hard-coded input image,
# compares them against every image in the pool, and returns
# the top K most similar matches
def retrieve_similar_images(input_path: str, queryset, top_k: int):
    input_feats = extract_all_features(input_path)
    results = []

    for photo in queryset:
        img_path = photo.image.path
        db_feats = load_or_compute_features(img_path)
        score, detail = compare_features(input_feats, db_feats)
        results.append((score, detail, photo))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

# display function arranges the input image and its top matches
# in a single row, annotates each with its similarity score,
# and renders them as a Matplotlib figure
# def display_top_matches(input_path: str, matches: list):
#     # 1) Show images side-by-side
#     images = [cv2.imread(input_path)] + [cv2.imread(m[2]["path"]) for m in matches]
#     titles = ["Input"] + [f"Match {i+1}\nScore {m[0]:.2f}" for i,m in enumerate(matches)]
#     plt.figure(figsize=(4*len(images), 4))
#     for idx, img in enumerate(images):
#         ax = plt.subplot(1, len(images), idx+1)
#         ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         ax.set_title(titles[idx])
#         ax.axis("off")
#     plt.tight_layout()
#     plt.show()

#     # 2) Print per-method metrics and formula
#     print("\nSimilarity Metrics Breakdown:")
#     for i, (score, detail, feats) in enumerate(matches, 1):
#         print(f"\nMatch #{i}: {feats['path']}")
#         for method, mscore in detail.items():
#             print(f"  {method:8s}: {mscore:.4f}")
#         print(f"  --> Overall similarity = (hist + ssim + orb + templ + edges + contours) / 6")
#         print(f"                         = {score:.4f}")

# if __name__ == "__main__":
#     matches = retrieve_similar_images(INPUT_IMAGE_PATH, IMAGE_POOL_DIR, TOP_K)
#     display_top_matches(INPUT_IMAGE_PATH, matches)