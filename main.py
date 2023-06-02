import sys
from sift import *
import numpy as np
from PIL import Image
import math

def main():
    args = sys.argv
    image_path = args[1]

    print("SIFT: " + image_path)
    image = Image.open(image_path)
    print("Opened image successfully.")
    log_file = open("log", "w")
    print("Creating SIFT model.")
    params = {}
    #params["dog_threshold"] = 5.0
    gray = False
    blur = False
    print("Non default params used:")
    for (key, value) in params.items():
        print(f"{key}: {value}")
    print("Image already in grayscale? ", gray)
    print("Image already pre-blurred? ", blur)
    sift = SIFT(image=image,
                params=params,
                gray=gray,
                blur=blur)
    sift.write_log(log_file, "ALGORITHM 1 ======================================")
    print("Algorithm 1: Computation of digital scale space") 
    octaves = sift.generate_scale_space(log_file=log_file)
    sift.write_log(log_file, "ALGORITHM 3 ======================================")
    print("Algorithm 3: Computation of the difference of Gaussians scale-space")
    diff = sift.generate_difference_space(octaves=octaves, log_file=log_file)
    sift.write_log(log_file, "ALGORITHM 4 ======================================")
    print("Algorithm 4: Scanning for discrete extrema")
    # SKIP EXTREMA ENABLED
    extrema = sift.scan_discrete_extrema(diff=diff, log_file=log_file)
    #extrema = sift.read_extrema_from_file(filename=image_path + ".extrema")
    #print("Writing extrema to file")
    #sift.write_extrema_to_file(extrema=extrema, filename=image_path + ".extrema")
    sift.write_log(log_file, "ALGORITHM 6 ======================================")
    print("Algorithm 6: Keypoint interpolation of extrema")
    extrema = sift.keypoints_interpolation(extrema=extrema, diff=diff, log_file=log_file)
    sift.write_log(log_file, "ALGORITHM 8 ======================================")
    print("Algorithm 8: Discard low contrast extrema")
    extrema = sift.discard_low_contrast(extrema=extrema, log_file=log_file)

    print("Algorithm 10: Compute gradient")
    gradients = sift.compute_gradient_at_scale_space(scale_space=octaves, log_file=log_file)

    log_file.close()
    extrema = sift.top_n_extrema(extrema, 200)
    img = np.array(image)
    for e in extrema:
        (o,s,m,n,sigma,x,y,omega) = e
#        if s < 0:
#            print(s)
        for i in range(0, 360, 5):
            offset_x = int(round(math.cos(i * math.pi / 180)*sigma*math.sqrt(2)*6))
            offset_y = int(round(math.sin(i * math.pi / 180)*sigma*math.sqrt(2)*6))
            if y + offset_y < img.shape[0] - 1 and x + offset_x < img.shape[1] - 1:
                img[int(y) + offset_y][int(x) + offset_x] = [255, 0, 0]




    img = Image.fromarray(np.uint8(img)).show()
    #sift.display_extrema(extrema)

    print("Algorithm 11: Compute reference orientations")
    extrema = sift.compute_reference_orientation(extrema=extrema, scale_space=octaves, gradient=gradients)
    for e in extrema:
        (o,s,x,y,sigma,theta) = e
        #print(x, y, sigma, theta)

    print("Algorithm 12: Compute feature descriptors")
    features = sift.construct_keypoint_descriptors(keypoints=extrema, gradient=gradients, scale_space=octaves)
    print("Completed SIFT.")

if __name__ == "__main__":
    main()
