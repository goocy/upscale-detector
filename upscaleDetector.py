# This code was written by goocy and is licensed under Creative Commons BY-NC 3.0.
# https://creativecommons.org/licenses/by-nc/3.0/
# Commercial use is prohibited.

# core idea: https://github.com/0x09/resdet
# other helpful sources:
# https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
# https://stackoverflow.com/questions/13904851/use-pythons-scipy-dct-ii-to-do-2d-or-nd-dct
# sliding median: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html
# hypothesis testing: https://stackoverflow.com/questions/48705448/z-score-calculation-from-mean-and-st-dev
# ffmpeg: https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md
# ffmpeg syntax: https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md#convert-sound-to-raw-pcm-audio
# unused sources:
# https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html
# https://pywavelets.readthedocs.io/en/latest/ref/2d-decompositions-overview.html

import matplotlib.pyplot as plt
import skimage.transform
import scipy.fftpack
import scipy.ndimage
import scipy.stats
import numpy as np
import argparse
import random
import ffmpeg # the package is called ffmpeg-python
import tqdm
import cv2

# https://stackoverflow.com/questions/38332642/plot-the-2d-fft-of-an-image
def dctn(x, norm="ortho"):
    for i in range(x.ndim):
        x = scipy.fftpack.dct(x, axis=i, norm=norm)
    return x

parser = argparse.ArgumentParser()
parser.add_argument('f')
args = parser.parse_args()
videoFilename = args.f
sampleCount = 200

# figure out metadata
probe = ffmpeg.probe(videoFilename)
video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
width = int(video_info['width'])
height = int(video_info['height'])
duration = int(float(probe['format']['duration']) * 1000)
print('Original resolution: {:d}x{:d}'.format(width, height))
videoShape = [width, height]
videoRatio = width / height

# extract random frames
timestamps = random.choices(range(duration-100), k=sampleCount)
xCurves = [0,]*sampleCount
yCurves = [0,]*sampleCount
print('Sampling random frames from the video...')
for i, timestamp in enumerate(tqdm.tqdm(timestamps)):
	imageBuffer, _ = (
	    ffmpeg.input(videoFilename, ss='{:.0f}'.format(timestamp/1000))
	    .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24', loglevel="quiet")
	    .run(capture_stdout=True)
	)
	if len(imageBuffer) > 0:
		image = np.frombuffer(imageBuffer, np.uint8).reshape([height, width, 3])
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		dct = dctn(gray)
		squaredDCT = np.power(dct,2)
		xDCT = np.sum(squaredDCT, axis=0)
		yDCT = np.sum(squaredDCT, axis=1)
		xDCT[xDCT == 0] = np.min(xDCT) / 10
		yDCT[yDCT == 0] = np.min(yDCT) / 10
		xCurves[i] = np.log10(xDCT)
		yCurves[i] = np.log10(yDCT)

# analyze peaks
xCurve = np.mean(xCurves, axis=0)
yCurve = np.mean(yCurves, axis=0)
rawCurves = [xCurve, yCurve]
spikyCurves = []
for axis in [0,1]:
	end = videoShape[axis]
	rawCurve = rawCurves[axis]
	start = round(end / 3) # only look below 3x upscaling
	smoothCurve = scipy.ndimage.median_filter(rawCurve, size=9, mode='reflect')
	spikyCurve = rawCurve - smoothCurve
	spikyCurves.append(spikyCurve)

	#plt.plot(spikyCurve)
	#plt.xlim([start, end])
	#amplitude = min(spikyCurve[start:])*1.1
	#plt.ylim(amplitude, -amplitude)
	#plt.show()

# combine the two curves
peakCount = 5
squeezedCurve = skimage.transform.resize(spikyCurves[0], (height,), preserve_range=True, mode='constant')
combinedCurve = spikyCurves[1] + squeezedCurve
minimumIndices = np.argsort(combinedCurve)[:peakCount]
peakConfidences = np.zeros((peakCount))
for i, minimumIndex in enumerate(minimumIndices):
	minimumValue = combinedCurve[minimumIndex]
	confidence = 1-scipy.stats.norm(np.mean(combinedCurve), np.std(combinedCurve)).cdf(minimumValue)
	peakConfidences[i] = confidence

# extract the most likely original video resolutions
if any(peakConfidences > 0.8):
	print('Most likely original resolutions:')
	for i, confidence in enumerate(peakConfidences):
		if confidence > 0.8:
			yIndex = minimumIndices[i]
			xIndex = yIndex * videoRatio
			confidenceText = '{:.3f}%'.format(confidence*100)
			print('x: {:.0f}, y: {:.0f} ({})'.format(xIndex, yIndex, confidenceText))
else:
	print('Video is most likely not upscaled.')