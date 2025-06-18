import cv2
import os
import uuid
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from django.conf import settings


def save_plot(fig, name_prefix="plot"):
    """Save Matplotlib figure and return relative path"""
    filename = f"{name_prefix}_{uuid.uuid4().hex}.png"
    relative_path = f"quality/{filename}"
    full_path = os.path.join(settings.MEDIA_ROOT, relative_path)

    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    fig.savefig(full_path, bbox_inches="tight")
    plt.close(fig)
    
    return settings.MEDIA_URL + relative_path


def assess_quality(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = gray.shape

    # ==================== METRIC FUNCTIONS ====================
    def defocus_blur(g):
        # How blurry is the picture? Lower = dreamy blur, higher = sharp focus
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    def motion_blur_proxy(g):
        # A quick way to sniff out motion lines: compare edge energy before/after Gaussian blur
        base = (np.mean(np.abs(cv2.Sobel(g,cv2.CV_64F,1,0))) +
                np.mean(np.abs(cv2.Sobel(g,cv2.CV_64F,0,1))))
        energies=[]
        for k in [(31,1),(1,31)]:
            b = cv2.GaussianBlur(g, k, 0)
            e = (np.mean(np.abs(cv2.Sobel(b,cv2.CV_64F,1,0))) +
                np.mean(np.abs(cv2.Sobel(b,cv2.CV_64F,0,1))))
            energies.append(e)
        # If the blurred edges lose a lot of energy, there's motion blur lurking
        return float(min(energies)/(base+1e-8))

    def dof_blur_ratio(g):
        # Simulate shallow depth of field: compare center edges vs. border edges
        edge = np.abs(cv2.Laplacian(g, cv2.CV_64F))
        c = edge[h//4:3*h//4, w//4:3*w//4].mean()
        b = np.concatenate([
            edge[:h//8,:].ravel(), edge[-h//8:,:].ravel(),
            edge[:,:w//8].ravel(),  edge[:,-w//8:].ravel()
        ]).mean()
        return float(c/(b+1e-8))

    def noise_flat(g, win=32, step=16):
        # Peek at "flat" patches to estimate sensor noise
        vs=[]; H,W=g.shape
        for y in range(0, H-win+1, step):
            for x in range(0, W-win+1, step):
                p = g[y:y+win, x:x+win]
                if p.std()<10: vs.append(p.var())
        return float(np.median(vs)) if vs else 0.0

    def clipping_metrics(img):
        # Count how many pixels are pinned at black or white for each channel
        out={}
        for i,ch in enumerate(("B","G","R")):
            out[ch] = (
                float(np.mean(img[:,:,i]==0)*100),
                float(np.mean(img[:,:,i]==255)*100)
            )
        return out

    def midtone_balance(g):
        # What percent of pixels sit in the comfy middle (neither too dark nor too bright)?
        return float(np.sum((g>=64)&(g<=192))/g.size)

    def local_dynamic_range(g, win=32, step=16):
        # Average contrast range in small patches
        vals=[]; H,W=g.shape
        for y in range(0, H-win+1, step):
            for x in range(0, W-win+1, step):
                p = g[y:y+win, x:x+win]
                vals.append((p.max()-p.min())/255)
        return float(np.mean(vals))

    def global_contrast(g):
        # Overall contrast: standard deviation of intensities
        return float(g.std()/255)

    def local_contrast_rms(g, win=32, step=16):
        # RMS contrast normalized by mean in patches (Weber contrast proxy)
        vals=[]; H,W=g.shape
        for y in range(0, H-win+1, step):
            for x in range(0, W-win+1, step):
                p = g[y:y+win, x:x+win].astype(np.float32)
                if p.mean()>1:
                    vals.append(p.std()/p.mean())
        return float(np.mean(vals)) if vals else 0.0

    def grayworld_cast(img):
        # Check for color cast: difference from gray world assumption
        avg = img.mean(axis=(0,1))/255; gw=avg.mean()
        return [float(c-gw) for c in avg]

    def saturation_stats(hsv):
        # How colorful is the scene? Mean and spread of saturation
        s = hsv[:,:,1]/255
        return float(s.mean()), float(s.std())
    
    def vignetting_metric(g):
        # Center vs corner brightness ratio for vignetting
        c  = g[h//4:3*h//4, w//4:3*w//4].mean()
        cr = np.concatenate([
            g[:h//4, :w//4].ravel(),  g[:h//4, -w//4:].ravel(),
            g[-h//4:, :w//4].ravel(), g[-h//4:, -w//4:].ravel()
        ]).mean()
        return float(cr/(c+1e-8))

    def hot_pixels_metric(g, thr=250):
        # Count tiny bright specks (stuck hot pixels)
        mask = g > thr
        n,_,stats,_ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        return sum(1 for a in stats[1:,cv2.CC_STAT_AREA] if a<3)

    metrics = {
        "Defocus Blur": defocus_blur(gray),
        "Motion Blur Proxy": motion_blur_proxy(gray),
        "DOF Blur Ratio": dof_blur_ratio(gray),
        "Noise Variance": noise_flat(gray),
        "Clipping": clipping_metrics(img),
        "Midtone Balance %": midtone_balance(gray) * 100,
        "Local Dynamic Range": local_dynamic_range(gray),
        "Global Contrast": global_contrast(gray),
        "Local RMS Contrast": local_contrast_rms(gray),
        "Grayworld Cast (B,G,R)": grayworld_cast(img),
        "Saturation (Mean, Std)": saturation_stats(hsv),
        "Vignetting": vignetting_metric(gray),
        "Hot Pixels Count": hot_pixels_metric(gray)
    }

    # ========== Example Visualizations ===============
    # ========== PER-PIXEL MAPS FOR VISUALIZATION =====
    WIN  = 32
    lap  = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    # Blur map: darker where sharp, brighter where blurry
    blur_map  = 1 - cv2.normalize(cv2.blur(lap,(WIN,WIN)), None, 0,1,cv2.NORM_MINMAX)

    sq        = gray.astype(np.float32)**2
    mean      = cv2.blur(gray.astype(np.float32),(WIN,WIN))
    # Estimate noise variance per patch and normalize
    var_map    = cv2.blur(sq,(WIN,WIN)) - mean**2
    noise_map  = cv2.normalize(var_map, None, 0,1,cv2.NORM_MINMAX)

    # Local contrast: STD/Mean
    std_map    = np.sqrt(cv2.blur((gray.astype(np.float32)-mean)**2,(WIN,WIN)))
    ctr_map    = cv2.normalize(std_map/(mean+1e-8), None, 0,1,cv2.NORM_MINMAX)

    # Vignetting fall-off effect
    Y,X        = np.indices((h,w))
    r          = np.sqrt((X-w/2)**2+(Y-h/2)**2)/np.sqrt((w/2)**2+(h/2)**2)
    vign_map   = cv2.normalize(cv2.normalize(gray.astype(np.float32),None,0,1,cv2.NORM_MINMAX)*(1-r), None,0,1,cv2.NORM_MINMAX)

    # Blockiness heat: highlight grid artifacts
    heat       = np.zeros_like(gray, dtype=np.float32)
    bs         = 8
    for y in range(bs, h, bs):
        heat[y-1,:] = np.abs(gray[y,:].astype(np.float32) - gray[y-1,:].astype(np.float32))
    for x in range(bs, w, bs):
        heat[:,x-1] = np.maximum(heat[:,x-1],
                                np.abs(gray[:,x].astype(np.float32)-gray[:,x-1].astype(np.float32)))
    block_heat = cv2.normalize(heat, None, 0,1,cv2.NORM_MINMAX)
    
    # ========= VISUALIZE ==========
    visuals = {}

    fig = plt.figure(figsize=(8,6))
    plt.title("Blur Heatmap")
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    blur_map = 1 - cv2.normalize(cv2.blur(lap, (32, 32)), None, 0, 1, cv2.NORM_MINMAX)
    plt.imshow(gray, cmap="gray")
    plt.imshow(blur_map, cmap="plasma", alpha=0.5, vmin=0, vmax=1)
    plt.colorbar(shrink=0.8, label="Blur Intensity")
    plt.axis("off")
    visuals["Blur Heatmap"] = save_plot(fig, "blur_heatmap")

    fig = plt.figure(figsize=(6,4))
    plt.title("Noise Heatmap: grainy bits exposed")
    plt.imshow(noise_map, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Normalized variance")
    plt.axis("off");
    visuals["Noise Heatmap"] = save_plot(fig, "noise_heatmap")

    fig = plt.figure(figsize=(6,4))
    plt.title("Exposure Histogram: darks vs brights showdown")
    plt.hist(gray.ravel(), bins=50, range=(0,255), color="gray")
    plt.axvline(0, linestyle="--", color="blue",  label="Under-exposed")
    plt.axvline(255, linestyle="--", color="red",  label="Over-exposed")
    plt.legend(); plt.xlabel("Intensity"); plt.ylabel("Frequency");
    visuals["Exposure Histogram"] = save_plot(fig, "exposure_hist")

    fig = plt.figure(figsize=(6,4))
    plt.title("Local RMS Contrast: patchy pop")
    plt.imshow(ctr_map, cmap="magma", vmin=0, vmax=1)
    plt.colorbar(label="STD/Mean")
    plt.axis("off");
    visuals["Local RMS Contrast"] = save_plot(fig, "local_contrast")

    # fig = plt.figure(figsize=(6,4))
    # plt.title("Gray-World Color Cast: color balance check")
    # plt.bar(["B","G","R"], [gwc[0],gwc[1],gwc[2]], color=["blue","green","red"])
    # plt.axhline(0, color="black"); plt.ylabel("Cast");
    # visuals.append(save_plot(fig, "color_balance"))

    fig = plt.figure(figsize=(6,4))
    plt.title("Blockiness Heatmap: grid artifacts at your service")
    plt.imshow(block_heat, cmap="inferno", vmin=0, vmax=1)
    plt.colorbar(label="Boundary diff")
    plt.axis("off");
    visuals["Blockiness Heatmap"] = save_plot(fig, "blockiness_heatmap")

    fig = plt.figure(figsize=(6,4))
    plt.title("Vignetting Fall-off: corners vs center brightness")
    plt.imshow(vign_map, cmap="inferno", vmin=0, vmax=1)
    plt.colorbar(label="Brightness")
    plt.axis("off");
    visuals["Vignetting Fall-off"] = save_plot(fig, "vignetting_fall")

    return metrics, visuals
