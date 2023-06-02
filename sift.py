from PIL import Image, ImageFilter, ImageOps, ImageChops
import numpy as np
import math
import time
from numpy.lib.stride_tricks import sliding_window_view

class SIFT:
    """
    Calculates SIFT feature descriptors for an input PIL "Image" object

    Parameters:
    sig_min : blur level of seed image
        default 0.8
    gamma_min : sampling distance in first image of first octave
        default 0.5 (interpolation of 2x)
    sig_in : initial blur level of input image
        default 0.5
    n_oct : number of octaves (limited by image size)
        default 8
    n_spo : number of scales per octave
        default 3
    dog_threshold : threshold over difference of gaussians response
        default 0.015
    c_edge : threshold over the ratio of principle curvatures
        default 10
    n_bins : number of bins in the gradient orientation histogram
        default 36
    lambda_ori : sets how local the analysis of the gradient distribution is
        gaussian window of standard deviation sigma * lambda
        patch width = 6 * lambda * sigma
        default 1.5
    lmt : threshold for considering local maxima in the gradient histogram
        default 0.8
    n_hist : number of histograms in the normalized patch is (nhist x nhist)
        default 4
    n_ori : number of bins in the descriptor histograms
        default 8
    lamba_desc : sets how local the descriptor is
        default 6

    """ 
    def __init__(self, image, params, gray = False, blur = False):
        """
        @params
        image : PIL Image Object
        params : dict(string, ___) to overwrite default parameters
        gray : Boolean: is the image already in grayscale?
        blur : Boolean: is the image already pre-blurred?
        """
        self.sig_min = 0.8
        self.gamma_min = 0.5
        self.sig_in = 0.5
        self.n_oct = 8
        self.n_spo = 3
        self.dog_threshold = 0.015
        self.c_edge = 10
        self.n_bins = 36
        self.lambda_ori = 1.5
        self.lmt = 0.8
        self.n_hist = 4
        self.n_ori = 8
        self.lambda_desc = 6

        for (key, value) in params.items():
            if key == "sig_min":
                self.sig_min = value
            elif key == "gamma_min":
                self.gamma_min = value
            elif key == "sig_in":
                self.sig_in = value
            elif key == "n_oct":
                self.n_oct = value
            elif key == "n_spo":
                self.n_spo = value
            elif key == "dog_threshold":
                self.dog_threshold = value
            elif key == "c_edge":
                self.c_edge = value
            elif key == "n_bins":
                self.n_bins = value
            elif key == "lambda_ori":
                self.lambda_ori = value
            elif key == "lmt":
                self.lmt = value
            elif key == "n_hist":
                self.n_hist = value
            elif key == "n_ori":
                self.n_ori = value
            elif key == "lambda_desc":
                self.lambda_desc = value
            else:
                raise("Invalid SIFT parameter: " + key)

        if not gray:
            image = ImageOps.grayscale(image)
        if not blur:
            image = image.filter(ImageFilter.GaussianBlur(radius = self.sig_in))
        self.image = image

    @staticmethod
    def bilinear_resample(image, gamma):
        """
        Perform bilinear sampling where gamma is the desired inter-pixel distance
        """
        size = (math.floor(image.width / gamma), math.floor(image.height / gamma))
        return image.resize(size=size, resample=Image.BILINEAR)

    @staticmethod
    def write_log(log_file, st):
        if not log_file is None:
            log_file.write(st + "\n")

    @staticmethod
    def write_extrema_to_file(extrema, filename):
        out = open(filename, "w")
        for e in extrema:
            (o,s,m,n) = e
            out.write(f"{o} {s} {m} {n}\n")
        out.close()

    @staticmethod
    def read_extrema_from_file(filename):
        in_file = open(filename, "r")
        extrema = []
        for line in in_file.readlines():
            elements = line.split()
            o = int(elements[0])
            s = int(elements[1])
            m = int(elements[2])
            n = int(elements[3])
            extrema.append((o,s,m,n))
        return extrema

    @staticmethod
    def sort_extrema(extrema):
        def key(e):
            (o,s,m,n,sigma,x,y,omega) = e
            return omega
        extrema.sort(reverse=True, key=key)

    @staticmethod
    def top_n_extrema(extrema, n):
        SIFT.sort_extrema(extrema)
        return extrema[:n]

    def display_extrema(self, extrema):
        img = np.asarray(self.image)
        for e in extrema:
            (o,s,m,n,sigma,x,y,omega) = e
            img[int(y)][int(x)] = 0
        img = Image.fromarray(np.uint8(img)).show()


    def generate_scale_space(self, log_file=None):
        """
        Generate the scale space using the input parameters
        @params:
        log: File descriptor to write log output
        @returns:
        Octave object
        """
        im = self.image # image can be assumed to be in grayscale and blurred
        SIFT.write_log(log_file, f"Input image size {im.width}*{im.height}")
        im = SIFT.bilinear_resample(im, self.gamma_min)
        SIFT.write_log(log_file, f"Image bilinear resampled by {self.gamma_min}")
        SIFT.write_log(log_file, f"New image size {im.width}*{im.height}")
        # create octaves
        octaves = [[im]] # octaves[0][0] is the starting image
        octaves.append([])
        # calculation of v(1)(0)
        sigma = (1.0 / self.gamma_min) * math.sqrt(self.sig_min**2 - self.sig_in**2)
        octaves[1].append(im.filter(ImageFilter.GaussianBlur(radius = sigma)))
        SIFT.write_log(log_file, f"First blur with sigma={sigma} for v(1)(0)")
        # calculate the other images in the first octave
        for s in range(1, self.n_spo + 3):
            sigma = (self.sig_min / self.gamma_min) * math.sqrt(2 ** (2 * s / self.n_spo) - 2 ** (2 * (s - 1) / self.n_spo))
            octaves[1].append(octaves[1][s - 1].filter(ImageFilter.GaussianBlur(radius=sigma)))
            SIFT.write_log(log_file, f"First octave scale {s} sigma={sigma} for v(1)({s})")
        
        # calculate the other octaves
        for o in range(2, self.n_oct + 1):
            octaves.append([])
            # subsample from the last of the previous octave
            octaves[o].append(SIFT.bilinear_resample(octaves[o - 1][self.n_spo], 2))
            SIFT.write_log(log_file, f"Generated first image in octave size {octaves[o][0].width}*{octaves[o][0].height}")
            for s in range(1, self.n_spo + 3):
                sigma = (self.sig_min / self.gamma_min) * math.sqrt(2 ** (2 * s / self.n_spo) - 2 ** (2 * (s - 1) / self.n_spo))
                octaves[o].append(octaves[o][s - 1].filter(ImageFilter.GaussianBlur(radius=sigma)))
                SIFT.write_log(log_file, f"Octave {o} scale {s} sigma={sigma} for v({o})({s})")

        return octaves

    
    def generate_difference_space(self, octaves, log_file=None):
        """
        Turn a given octave into its difference space equivalent by subtracting adjacent 
        scales.

        @param
        octave : output of generate_scale_space. octave[1..n_oct] are lists of images
        log_file : file to write any debugging output

        @return
        difference space in the same structure as the octaves
        """
        output = [[]]
        for o in range(1, self.n_oct + 1):
            output.append([])
            for s in range(0, self.n_spo + 2):
                output[o].append(ImageChops.difference(octaves[o][s + 1], octaves[o][s]))
                SIFT.write_log(log_file, f"Wrote difference between v({o})({s + 1}), v({o})({s}) to output")

        return output
                

    def scan_discrete_extrema(self, diff, log_file=None):
        """
        Scan over every pixel in all layers of the difference of Gaussians space.

        If any pixel is larger or smaller than its 26 neighbors, add to list of extrema
        @param
        diff : difference space generated by generate_difference_space()
        log_file : file to write output logs to

        @return
        list of {(o, s, m, n)} where m,n = x,y
        """
        maxes = [[]]
        mins = [[]]
        values = [[]]
        for o in range(1, self.n_oct + 1):
            maxes.append([])
            mins.append([])
            values.append([])
            for s in range(0, self.n_spo + 2):
                layer = np.array(diff[o][s])
                windows = sliding_window_view(layer, window_shape=(3,3))
                maxes[o].append(np.max(windows, axis=(2,3)))
                mins[o].append(np.min(windows, axis=(2,3)))
                values[o].append(layer)


        extrema = []
        #cutoff = 0.8 * self.dog_threshold
        for o in range(1, self.n_oct + 1):
            for s in range(1, self.n_spo + 1):
                # check cutoff
                # check extrema
                centers = values[o][s][1:-1,1:-1]
                #print(values[o][s].shape, centers.shape, maxes[o][s].shape)
                is_max = np.logical_and(centers == maxes[o][s], np.logical_and(centers > maxes[o][s-1], centers > maxes[o][s+1]))
                is_min = np.logical_and(centers == mins[o][s], np.logical_and(centers < mins[o][s-1], centers < mins[o][s+1]))
                is_threshold = (centers > self.dog_threshold)
                quality_point = np.logical_and(is_threshold, np.logical_or(is_max, is_min))
                for x in range(1, centers.shape[0]):
                    for y in range(1, centers.shape[1]):
                        if quality_point[x - 1][y - 1]:
                            extrema.append((o,s,y-1,x-1))

        return extrema


    def extrema_conservative_test(self, diff, extrema, log_file=None):
        """
        This test takes the extrema and does a basic filtering on their values.

        The threshold value is set as a parameter for the SIFT

        @param
        diff : difference space
        extrema : list of extrema (o,s,m,n)

        @returns
        list of extrema that pass this filter
        """
        output = []
        SIFT.write_log(log_file, f"Input extrema #: {len(extrema)}")
        SIFT.write_log(log_file, f"Filtering on w >= : {0.8 * self.dog_threshold}")
        for e in extrema:
            (o,s,m,n) = e
            value = diff[o][s].getpixel((m,n))
            if value >= 0.8 * self.dog_threshold:
                output.append(e)
                
        SIFT.write_log(log_file, f"Conservative test filtered extrema #: {len(output)}")
        return output


    def keypoints_interpolation(self, diff, extrema, log_file=None):
        """
        Run the interpolation algorithm on each extrema
        """
        def quadratic_interpolation(o,se,me,ne):
            g = np.zeros((3,1))
            h = np.zeros((3,3))
            g[0][0] = (diff[o][se + 1].getpixel((me, ne)) - diff[o][se + 1].getpixel((me, ne))) / 2
            g[1][0] = (diff[o][se].getpixel((me + 1, ne)) - diff[o][s].getpixel((me - 1, ne))) / 2
            g[2][0] = (diff[o][se].getpixel((me, ne + 1)) - diff[o][s].getpixel((me, ne - 1))) / 2
            
            h = np.zeros((3,3))
            h11 = diff[o][se + 1].getpixel((me, ne)) + diff[o][se - 1].getpixel((me, ne)) - 2*diff[o][se].getpixel((me, ne))
            h22 = diff[o][se].getpixel((me + 1, ne)) + diff[o][se].getpixel((me - 1, ne)) - 2*diff[o][se].getpixel((me, ne))
            h33 = diff[o][se].getpixel((me, ne + 1)) + diff[o][se].getpixel((me, ne - 1)) - 2*diff[o][se].getpixel((me, ne))
            h12 = (
                diff[o][se + 1].getpixel((me + 1, ne)) - 
                diff[o][se + 1].getpixel((me - 1, ne)) -
                diff[o][se - 1].getpixel((me + 1, ne)) +
                diff[o][se - 1].getpixel((me - 1, ne))
            ) / 4
            h13 = (
                diff[o][se + 1].getpixel((me, ne + 1)) - 
                diff[o][se + 1].getpixel((me, ne - 1)) -
                diff[o][se - 1].getpixel((me, ne + 1)) +
                diff[o][se - 1].getpixel((me, ne - 1))
            ) / 4
            h13 = (
                diff[o][se].getpixel((me + 1, ne + 1)) - 
                diff[o][se].getpixel((me + 1, ne - 1)) -
                diff[o][se].getpixel((me - 1, ne + 1)) +
                diff[o][se].getpixel((me - 1, ne - 1))
            ) / 4
            h[0][0] = h11
            h[1][0] = h12
            h[0][1] = h12
            h[0][2] = h12
            h[2][0] = h12
            h[1][1] = h22
            h[2][1] = h13
            h[1][2] = h13
            h[2][2] = h33
            if np.linalg.det(h) == 0:
                return None
            hinv = np.linalg.inv(h)
            alpha = -1 * np.matmul(hinv, g)
            omega_1 = 0.5*np.matmul(g.transpose(), alpha)
            omega = diff[o][se].getpixel((me, ne)) + omega_1[0][0]
            return (alpha, omega)


        output = []
        for e in extrema:
            (o_e, s_e, m_e, n_e) = e
            (s,m,n) = (s_e, m_e, n_e)
            SIFT.write_log(log_file, f"Extrema {o_e},{s},{m},{n}")
            iterations = 0
            # instantiate these in case this is a successful point
            sigma = 0
            while iterations < 5:
                iterations += 1
                SIFT.write_log(log_file, f"Iteration {iterations}")
                try:
                    qi = quadratic_interpolation(o_e,s,m,n)
                    if qi is None:
                        SIFT.write_log(log_file, "Quadratic failed. Singular matrix.")
                        break
                    (alpha, omega) = qi
                    gamma_o_e = ((2 ** (o_e - 1)) * self.gamma_min)
                    sigma = (gamma_o_e * self.gamma_min) * self.sig_min * 2**((alpha[0][0] + s)/self.n_spo)
                    x = gamma_o_e * (alpha[1][0] + m)
                    y = gamma_o_e * (alpha[2][0] + n)

                    SIFT.write_log(log_file, f"Comparing fields: o {o_e}:{gamma_o_e} gamma")
                    SIFT.write_log(log_file, f"m {m}:{x} x")
                    SIFT.write_log(log_file, f"n {n}:{y} y")
                    s = int(round(s + alpha[0][0]))
                    m = int(round(m + alpha[1][0]))
                    n = int(round(n + alpha[2][0]))
                    SIFT.write_log(log_file, f"New (s,m,n): {s},{m},{n}")

                    if max(alpha) < 0.6 and s >= 0 and s < self.n_spo:
                        output.append((o_e,s,m,n,sigma,x,y,omega))
                        SIFT.write_log(log_file, f"Passed: {o_e},{s},{m},{n},{sigma},{x},{y},{omega}")
                        break
                except:
                    break

        return output


    def discard_low_contrast(self, extrema, log_file=None):
        """
        Discard low contrast extrema based on their omega value from previous calculations
        """
        output = []
        SIFT.write_log(log_file, f"Number of extrema: {len(extrema)}")
        for e in extrema:
            (o,s,m,n,sigma,x,y,omega) = e
            if abs(omega) >= self.dog_threshold:
                output.append(e)

        SIFT.write_log(log_file, f"Number of extrema after filter: {len(output)}")
        return output


    def discard_edge_keypoints(self, extrema, diff, log_file=None):
        """
        Discard candidate keypoints on edges
        """
        def hessian(o,s,m,n):
            h11 = diff[o][s].getpixel((m+1,n)) + diff[o][s].getpixel((m-1,n)) - 2*diff[o][s].getpixel((m,n))
            h22 = diff[o][s].getpixel((m,n+1)) + diff[o][s].getpixel((m,n-1)) - 2*diff[o][s].getpixel((m,n))
            h12 = (diff[o][s].getpixel((m+1,n+1)) - 
                diff[o][s].getpixel((m+1,n-1)) -
                diff[o][s].getpixel((m-1,n+1)) +
                diff[o][s].getpixel((m-1,n-1))) / 4
            output = np.zeros((2,2))
            output[0][0] = h11
            output[1][0] = h12
            output[0][1] = h12
            output[1][1] = h22
            return output
        
        output = []
        SIFT.write_log(log_file, f"Number of extrema: {len(extrema)}")
        for e in extrema:
            try:
                (o,s,m,n,sigma,x,y,omega) = e
                h = hessian(o,s,m,n)
                edgeness = (h.trace()**2) / np.linalg.det(h)
                max_edgeness = ((self.c_edge + 1)**2) / self.c_edge
                if edgeness < max_edgeness:
                    output.append(e)
            except:
                continue

        SIFT.write_log(log_file, f"Number of extrema after edge removal: {len(output)}")
        return output


    def compute_gradient_at_scale_space(self, scale_space, log_file=None):
        output = [[]]
        for o in range(1, self.n_oct + 1):
            output.append([])
            for s in range(0, self.n_spo + 1):
                img = np.array(scale_space[o][s])
                gradient_m = (np.roll(img, 1, axis=0) - np.roll(img, -1, axis=0)) / 2
                gradient_n = (np.roll(img, 1, axis=1) - np.roll(img, -1, axis=1)) / 2
                output[o].append((gradient_m, gradient_n))

        return output
    
    def compute_reference_orientation(self, extrema, scale_space, gradient):
        output = []
        for keypoint in extrema:
            (o_key, s_key, _, _, sigma_key, x_key, y_key, omega) = keypoint
            if x_key < 3*self.lambda_ori*sigma_key:
                continue # too close to border
            if x_key > scale_space[0][0].height - 3*self.lambda_ori*sigma_key:
                continue
            if y_key < 3*self.lambda_ori*sigma_key:
                continue # too close to border
            if y_key > scale_space[0][0].width - 3*self.lambda_ori*sigma_key:
                continue
            # initialize orientation histogram
            h = [0.0]
            for k in range(1, self.n_bins + 1):
                h.append(0.0)

            gamma_o_e = ((2 ** (o_key - 1)) * self.gamma_min)
            m_low = int(round((x_key - 3*self.lambda_ori*sigma_key)/(gamma_o_e)))
            m_high = int(round((x_key + 3*self.lambda_ori*sigma_key)/(gamma_o_e)))
            n_low = int(round((y_key - 3*self.lambda_ori*sigma_key)/(gamma_o_e)))
            n_high = int(round((y_key + 3*self.lambda_ori*sigma_key)/(gamma_o_e)))
            if m_low < 0 or n_low < 0:
                continue
            if m_high >= gradient[o_key][s_key][0].shape[1]:
                continue
            if n_high >= gradient[o_key][s_key][0].shape[0]:
                continue
            for m in range(m_low, m_high + 1):
                for n in range(n_low, n_high + 1):
                    exponent = ((m*gamma_o_e) - x_key)**2 + ((n*gamma_o_e) - y_key)**2
                    exponent = exponent / (-2 * (self.lambda_ori * sigma_key)**2)
                    c = math.exp(exponent)
                    c = c * math.sqrt((gradient[o_key][s_key][0][n][m])**2 + (gradient[o_key][s_key][1][n][m])**2)
                    # bori
                    b_ori = math.atan2(gradient[o_key][s_key][0][n][m], gradient[o_key][s_key][1][n][m])
                    b_ori = b_ori % (2*math.pi)
                    b_ori *= self.n_bins / (2*math.pi)
                    h[int(round(b_ori))] += c

            # apply 6x circular convolution filter?

            theta_key = 0
            for k in range(1, self.n_bins + 1):
                try:
                    if h[k] < h[(k - 1) % self.n_bins]:
                        continue
                    if h[k] < h[(k + 1) % self.n_bins]:
                        continue
                    if h[k] < 0.8 * max(h):
                        continue
                    h_minus = h[(k - 1) % self.n_bins]
                    h_plus = h[(k + 1) % self.n_bins]
                    theta_key = (2*math.pi*(k - 1)) / self.n_bins
                    theta_key += (math.pi / self.n_bins)*((h_minus - h_plus)/(h_minus - 2*h[k] + h_plus))
                except:
                    continue

            output.append((o_key, s_key, x_key, y_key, sigma_key, theta_key))

        return output

        
    def construct_keypoint_descriptors(self, keypoints, scale_space, gradient):
        lambda_desc = self.lambda_desc
        output = []
        for keypoint in keypoints:
            (o_key,s_key,x_key,y_key,sigma_key,theta_key) = keypoint
            if x_key < (math.sqrt(2)*lambda_desc*sigma_key):
                continue
            if x_key > scale_space[0][0].height - (math.sqrt(2)*lambda_desc*sigma_key):
                continue
            if y_key < (math.sqrt(2)*lambda_desc*sigma_key):
                continue
            if y_key > scale_space[0][0].width - (math.sqrt(2)*lambda_desc*sigma_key):
                continue
            h = np.zeros((self.n_hist, self.n_hist, self.n_ori))
            gamma_o_e = ((2 ** (o_key - 1)) * self.gamma_min)
            const = (self.n_hist + 1) / self.n_hist
            m_low = int(round((x_key - (math.sqrt(2))*lambda_desc*sigma_key*const)/(gamma_o_e)))
            m_high = int(round((x_key + math.sqrt(2)*lambda_desc*sigma_key*const)/(gamma_o_e)))
            n_low = int(round((y_key - math.sqrt(2)*lambda_desc*sigma_key*const)/(gamma_o_e)))
            n_high = int(round((y_key + math.sqrt(2)*lambda_desc*sigma_key*const)/(gamma_o_e)))
            if m_low < 0 or n_low < 0:
                continue
            if m_high >= gradient[o_key][s_key][0].shape[1]:
                continue
            if n_high >= gradient[o_key][s_key][0].shape[0]:
                continue
            for m in range(m_low, m_high + 1):
                for n in range(n_low, n_high + 1):
                    x_mn = ((m*gamma_o_e - x_key)*math.cos(theta_key) + (n*gamma_o_e - y_key)*math.sin(theta_key)) / sigma_key
                    y_mn = (-(m*gamma_o_e - x_key)*math.sin(theta_key) + (n*gamma_o_e - y_key)*math.cos(theta_key)) / sigma_key
                    if max(x_mn, y_mn) < lambda_desc*const:
                        theta_mn = math.atan2(gradient[o_key][s_key][0][n][m], gradient[o_key][s_key][1][n][m])
                        theta_mn -= theta_key
                        theta_mn = theta_mn % (2 * math.pi)


                        exponent = ((m*gamma_o_e) - x_key)**2 + ((n*gamma_o_e) - y_key)**2
                        exponent = exponent / (-2 * (lambda_desc * sigma_key)**2)
                        c = math.exp(exponent)
                        c = c * math.sqrt((gradient[o_key][s_key][0][n][m])**2 + (gradient[o_key][s_key][1][n][m])**2)
                        for i in range(1, self.n_hist + 1):
                            for j in range(1, self.n_hist + 1):
                                x_i = (i - (1 + self.n_hist) / 2)*(2*lambda_desc / self.n_hist)
                                y_j = (j - (1 + self.n_hist) / 2)*(2*lambda_desc / self.n_hist)
                                if abs(x_i - x_mn) <= (2*lambda_desc / self.n_hist) and abs(y_j - y_mn) <= (2*lambda_desc / self.n_hist):
                                    for k in range(1, self.n_ori + 1):
                                        theta_k = (2*math.pi*(k-1))/self.n_bins
                                        if not abs(theta_k - theta_mn % (2*math.pi)) < (2*math.pi / self.n_ori):
                                            continue
                                        val = (1 - (self.n_hist / (2 * lambda_desc))*abs(x_mn - x_i)) 
                                        val *= (1 - (self.n_hist / (2 * lambda_desc))*abs(y_mn - y_j)) 
                                        val *= (1 - (self.n_ori / (2 * math.pi))*abs(theta_mn - theta_k % (2*math.pi)))
                                        val *= c
                                        h[i-1][j-1][k-1] = h[i-1][j-1][k-1] + val

            # build feature descriptor
            f = np.zeros((self.n_hist * self.n_hist * self.n_ori))
            for i in range(1, self.n_hist + 1):
                for j in range(1, self.n_hist + 1):
                    for k in range(1, self.n_ori + 1):
                        idx = (i - 1)*self.n_hist*self.n_ori + (j - 1)*self.n_ori + k
                        f[idx - 1] = h[i-1][j-1][k-1]

            norm = np.linalg.norm(f)
            for l in range(1, (self.n_hist * self.n_hist * self.n_ori) + 1):
                f_l = min(f[l - 1], 0.2*norm)
                f_l = min(512*f_l/norm, 255)

            output.append((x_key, y_key, sigma_key, theta_key, f))


                





